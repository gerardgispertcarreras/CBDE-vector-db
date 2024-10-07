import sys
from connect import connect
from postgres.script_1 import script_1
from postgres.script_2 import script_2
from postgres.script_3 import script_3

SCRIPTS = {
    "sql-script-1": script_1,
    "sql-script-2": script_2,
    "sql-script-3": script_3,
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
