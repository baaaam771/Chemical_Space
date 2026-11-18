"""Who is using this server.

The engine never needed this: one person, one laptop, one browser. Putting it
on a shared host changes two things at once, and only one of them is about
keeping strangers out.

The other is that **the save path is derived from the user**. Before this,
every browser posted ``{"name": "web"}`` and every save landed in
``data/sessions/web/`` -- so the second person to press 저장 destroyed the
first person's exploration tree, silently, with no error anywhere. Identity is
what fixes that, and it is needed on a trusted internal network just as much
as on the open internet. Treating "we are behind the firewall" as a reason to
skip it would bake "there is no such thing as a user" into the save logic, the
session lookup and the UI, which is exactly the part that is expensive to undo
later.

So: real identity now, with the network boundary left to the network.

What this is
    A small user file with scrypt-hashed passwords, and a signed cookie.
What this is not
    A user directory, password reset, roles, or anything self-service. Adding
    a colleague is ``scripts/manage_users.py add <name>``.

Security notes, since this is the file where getting it wrong matters:

* Passwords are hashed with :func:`hashlib.scrypt`, which is memory-hard, and
  never stored or logged in the clear.
* Every comparison of a secret goes through :func:`hmac.compare_digest`, so a
  wrong password cannot be found a byte at a time by timing the reply.
* The cookie is an HMAC over ``name|expiry``. The browser can read its own
  name out of it, which is harmless; it cannot forge one without the server
  secret, and it cannot extend the expiry because the expiry is signed.
* The cookie is ``HttpOnly``, so page scripts cannot read it, and ``SameSite=Lax``,
  so another site cannot ride it.
* ``Secure`` is **off** by default because the first deployment is plain HTTP
  on an internal network, where a ``Secure`` cookie would simply never be
  sent. Turn it on with ``--secure-cookie`` the moment there is TLS in front.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = [
    "AuthError",
    "UserStore",
    "COOKIE_NAME",
    "NAME_PATTERN",
    "valid_name",
]

#: Cookie the browser carries. Named rather than generic so it is obvious in
#: devtools which application set it.
COOKIE_NAME = "chemspace_user"

#: How long a login lasts. Long enough that a working day never interrupts
#: itself, short enough that a forgotten browser on a shared machine stops
#: working within the week.
SESSION_TTL = 7 * 24 * 3600

#: Letters, digits, underscore and hyphen; must start with a letter or digit.
#:
#: Dots are excluded on purpose. A user name becomes a **directory name** under
#: ``data/sessions/``, and the moment dots are allowed, ``..`` is a legal name
#: and the save path escapes the directory it was supposed to stay in. The old
#: save-name pattern ``^[A-Za-z0-9._-]{1,64}$`` had exactly that hole -- ``..``
#: matched it. Excluding dots closes it at the grammar rather than relying on
#: anyone remembering to check; :func:`safe_subdir` then checks anyway.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"
_NAME_RE = re.compile(NAME_PATTERN)

#: scrypt parameters. n=2**14 with r=8 costs about 16 MB and ~50 ms per
#: attempt here, which is unnoticeable on a login and ruinous for a brute
#: force. They are stored beside each hash so they can be raised later without
#: invalidating existing passwords.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
DKLEN = 32


class AuthError(Exception):
    """Login refused. The message is safe to show the user."""


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def safe_subdir(root: Path, *parts: str) -> Path:
    """``root`` joined with ``parts``, refusing anything that escapes ``root``.

    The second line of defence behind :data:`NAME_PATTERN`. It resolves the
    result and checks containment, so a name that gets through validation --
    through a future pattern change, a symlink, or a Windows device name --
    still cannot write outside the sessions directory.
    """
    root = Path(root).resolve()
    for part in parts:
        if not valid_name(part):
            raise AuthError(f"이름으로 쓸 수 없는 값입니다: {part!r}")
    target = root.joinpath(*parts).resolve()
    if target != root and root not in target.parents:
        raise AuthError("저장 경로가 세션 디렉터리를 벗어납니다")
    return target


def hash_password(password: str) -> dict:
    """A fresh salt and an scrypt hash, as a JSON-storable record."""
    if len(password) < 8:
        raise AuthError("비밀번호는 8자 이상이어야 합니다")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        dklen=DKLEN, maxmem=64 * 1024 * 1024,
    )
    return {
        "salt": salt.hex(),
        "hash": digest.hex(),
        "n": SCRYPT_N,
        "r": SCRYPT_R,
        "p": SCRYPT_P,
        "dklen": DKLEN,
    }


def verify_password(password: str, record: dict) -> bool:
    """Constant-time check of ``password`` against a stored record."""
    try:
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(record["salt"]),
            n=int(record["n"]), r=int(record["r"]), p=int(record["p"]),
            dklen=int(record["dklen"]),
            maxmem=64 * 1024 * 1024,
        )
    except (KeyError, ValueError) as exc:  # a corrupt record is not a login
        log.warning("unreadable password record (%s)", exc)
        return False
    return hmac.compare_digest(digest.hex(), str(record.get("hash", "")))


@dataclass
class UserStore:
    """Users on disk, plus the secret that signs their cookies.

    ``users_path`` is JSON: ``{"users": {"<name>": {<scrypt record>}}}``.
    ``secret_path`` holds 32 random bytes, hex, created on first use. Deleting
    it logs everybody out, which is the way to revoke every session at once.
    """

    users_path: Path
    secret_path: Path

    def __post_init__(self) -> None:
        self.users_path = Path(self.users_path)
        self.secret_path = Path(self.secret_path)

    # ------------------------------------------------------------ user file

    def _read(self) -> dict:
        if not self.users_path.exists():
            return {"users": {}}
        try:
            data = json.loads(self.users_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AuthError(f"사용자 파일을 읽을 수 없습니다: {exc}") from None
        data.setdefault("users", {})
        return data

    def _write(self, data: dict) -> None:
        self.users_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.users_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.users_path)
        # Best effort: on Windows this is close to a no-op, but on the Linux
        # box this file ends up on it keeps the hashes off other accounts.
        try:
            os.chmod(self.users_path, 0o600)
        except OSError:
            pass

    def names(self) -> list[str]:
        return sorted(self._read()["users"])

    def __len__(self) -> int:
        return len(self._read()["users"])

    def add(self, name: str, password: str, replace: bool = False) -> None:
        if not valid_name(name):
            raise AuthError(
                f"사용자 이름은 영문/숫자로 시작하고 영문·숫자·_·- 만 쓸 수 있습니다: {name!r}"
            )
        data = self._read()
        if name in data["users"] and not replace:
            raise AuthError(f"이미 있는 사용자입니다: {name}")
        data["users"][name] = hash_password(password)
        self._write(data)

    def remove(self, name: str) -> bool:
        data = self._read()
        if data["users"].pop(name, None) is None:
            return False
        self._write(data)
        return True

    def check(self, name: str, password: str) -> bool:
        """True when ``name``/``password`` match a stored user.

        An unknown name still pays for one scrypt run against a throwaway
        record, so "no such user" and "wrong password" take the same time and
        the reply cannot be used to enumerate who has an account.
        """
        record = self._read()["users"].get(name)
        if record is None:
            verify_password(password, hash_password("x" * 12))
            return False
        return verify_password(password, record)

    # --------------------------------------------------------------- secret

    def secret(self) -> bytes:
        if self.secret_path.exists():
            return bytes.fromhex(self.secret_path.read_text(encoding="utf-8").strip())
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_bytes(32)
        self.secret_path.write_text(value.hex(), encoding="utf-8")
        try:
            os.chmod(self.secret_path, 0o600)
        except OSError:
            pass
        log.info("created a new cookie secret at %s", self.secret_path)
        return value

    # --------------------------------------------------------------- tokens

    def issue(self, name: str, ttl: int = SESSION_TTL) -> str:
        """``<name>.<expiry>.<hmac>`` -- the cookie value."""
        expiry = int(time.time() + ttl)
        payload = f"{name}.{expiry}"
        sig = hmac.new(self.secret(), payload.encode(), hashlib.sha256).digest()
        return f"{payload}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"

    def identify(self, token: str | None) -> str | None:
        """The user a cookie proves, or ``None``.

        Returns ``None`` for every kind of failure -- malformed, expired,
        forged, or naming a user who has since been removed -- because the
        caller's only sensible response to any of them is the login page.
        """
        if not token:
            return None
        try:
            name, expiry_s, _ = token.split(".", 2)
        except ValueError:
            return None
        if not valid_name(name):
            return None
        try:
            expiry = int(expiry_s)
        except ValueError:
            return None
        if expiry < time.time():
            return None
        # Re-issue from the claimed payload and compare whole tokens: this
        # checks the signature and the expiry together, so a token cannot be
        # replayed with a later expiry stitched on.
        payload = f"{name}.{expiry}"
        sig = hmac.new(self.secret(), payload.encode(), hashlib.sha256).digest()
        expected = f"{payload}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"
        if not hmac.compare_digest(token, expected):
            return None
        if name not in self._read()["users"]:
            return None
        return name
