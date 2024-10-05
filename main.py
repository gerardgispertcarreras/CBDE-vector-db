import sys
from connect import connect
from postgres.script_1 import script_1

SCRIPTS = {
    "sql-script-1": script_1,
}


def usage():
    print("Usage: python main.py <script>")
    print(f"Where script can take the values: {SCRIPTS.keys()}")
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        usage()
    script = sys.argv[1]
    connect(SCRIPTS[script])


if __name__ == "__main__":
    main()
