"""Add, list and remove the people who may use the web UI.

    python scripts/manage_users.py add     <name>     # prompts twice, hidden
    python scripts/manage_users.py add     <name> --generate
    python scripts/manage_users.py passwd  <name>
    python scripts/manage_users.py list
    python scripts/manage_users.py remove  <name>

Deliberately not a web page. Self-service account creation is a whole
surface -- invitations, verification, lockout, recovery -- and the actual
requirement is "five colleagues in one lab", which a command and a file meet
completely.

The password is read with :func:`getpass.getpass`, so it is never echoed,
never becomes a command-line argument (which any user on the host can read
out of the process list), and never reaches the shell history. ``--generate``
is the better option where the host can be trusted to print it once.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chemspace.data.chembl import resolve_data_path  # noqa: E402
from chemspace.web.auth import AuthError, UserStore  # noqa: E402

#: No look-alike characters: a generated password usually gets read off a
#: screen and typed somewhere else, and 0/O or l/1 turns that into a support
#: question.
ALPHABET = (
    "".join(c for c in string.ascii_letters + string.digits if c not in "0O1lI")
)


def generate(length: int = 16) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def read_password(name: str) -> str:
    first = getpass.getpass(f"{name}의 비밀번호 (8자 이상): ")
    second = getpass.getpass("한 번 더: ")
    if first != second:
        raise AuthError("두 입력이 다릅니다")
    return first


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--users", default="data/users.json", help="user file (default: data/users.json)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="새 사용자")
    p_add.add_argument("name")
    p_add.add_argument("--generate", action="store_true", help="비밀번호를 만들어 한 번 출력")

    p_pw = sub.add_parser("passwd", help="비밀번호 변경")
    p_pw.add_argument("name")
    p_pw.add_argument("--generate", action="store_true")

    sub.add_parser("list", help="사용자 목록")

    p_rm = sub.add_parser("remove", help="사용자 삭제")
    p_rm.add_argument("name")

    args = ap.parse_args()
    users_path = resolve_data_path(args.users)
    store = UserStore(users_path, users_path.parent / "cookie_secret")

    try:
        if args.cmd == "list":
            names = store.names()
            if not names:
                print(f"등록된 사용자가 없습니다 ({users_path})")
                return 0
            print(f"{users_path} — {len(names)}명")
            for n in names:
                print(f"  {n}")
            return 0

        if args.cmd == "remove":
            if not store.remove(args.name):
                print(f"그런 사용자가 없습니다: {args.name}", file=sys.stderr)
                return 1
            print(f"삭제했습니다: {args.name}")
            print("  (이미 로그인한 브라우저도 즉시 접근이 끊깁니다)")
            return 0

        replace = args.cmd == "passwd"
        if replace and args.name not in store.names():
            print(f"그런 사용자가 없습니다: {args.name}", file=sys.stderr)
            return 1

        if args.generate:
            password = generate()
        else:
            password = read_password(args.name)

        store.add(args.name, password, replace=replace)
        print(f"{'비밀번호를 바꿨습니다' if replace else '추가했습니다'}: {args.name}")
        if args.generate:
            print(f"  비밀번호: {password}")
            print("  이 줄은 다시 볼 수 없습니다. 지금 전달하세요.")
        print(f"  파일: {users_path}")
    except AuthError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n취소했습니다", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
