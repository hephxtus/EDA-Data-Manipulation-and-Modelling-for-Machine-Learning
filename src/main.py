from src import Part1, Part2, Part3

def main():
    parts = [Part1, Part2, Part3]
    ex_bbol = False
    while not ex_bbol:
        index = int(input("Part to run ({}), 0 to exit or {} to run all: ".format(len(parts), len(parts) + 1)))

        if index == 0:
            ex_bbol = True
            exit()
        elif index > len(parts):
            for part in parts:
                print("Running {}".format(part.__name__))
                part.main()
        else:
            parts[index - 1].main()

if __name__ == '__main__':
    main()