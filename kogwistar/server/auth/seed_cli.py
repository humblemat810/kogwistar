from __future__ import annotations
import argparse
import sys
import os
from .db import create_auth_engine, init_auth_db, get_session
from .seeding import seed_auth_data


def main():
    parser = argparse.ArgumentParser(
        description="Seed/Upsert Auth data into the database."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("AUTH_DB_URL", "sqlite:///auth.sqlite"),
        help="Database URL (default: sqlite:///auth.sqlite)",
    )
    parser.add_argument("--file", "-f", help="Path to JSON seed file")
    parser.add_argument("--json", "-j", help="JSON string to seed")

    args = parser.parse_args()

    # Initialize DB
    engine = create_auth_engine(args.db_url)
    init_auth_db(engine)

    seed_json = None
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r") as f:
            seed_json = f.read()
    elif args.json:
        seed_json = args.json

    session = get_session()
    try:
        seed_auth_data(session, seed_json=seed_json)
        session.commit()
        print("Successfully seeded auth data.")
    except Exception as e:
        session.rollback()
        print(f"Error seeding data: {e}")
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    main()
