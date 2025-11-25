import random

def bubble_sort(list_to_sort, ascending='y'):
    n = len(list_to_sort)
    for i in range(n - 1):
        swapped = False

        if ascending == 'y':
            for j in range(n - i - 1):
                if list_to_sort[j] > list_to_sort[j + 1]:
                    list_to_sort[j], list_to_sort[j + 1] = list_to_sort[j + 1], list_to_sort[j] # Swap over if j is greater than j + 1
                    swapped = True
        else:
            for j in range(n - i - 1):
                if list_to_sort[j] < list_to_sort[j + 1]:
                    list_to_sort[j], list_to_sort[j + 1] = list_to_sort[j + 1], list_to_sort[j] # Swap over if j is greater than j + 1
                    swapped = True
        if not swapped:
            list_sorted = list_to_sort
            return list_sorted


def main():
    n = int(input("How many numbers would you like to sort?: "))
    choice = input("Would you like to enter the numbers yourself? (y/n): ")
    ascending = input("Would you like the numbers ascending? (n for descending) (y/n): ")

    if choice.lower() == 'y':
        list_to_sort = []
        print(f"Enter {n} numbers:")
        for _ in range(n):
            num = int(input("> "))
            list_to_sort.append(num)
    else:
        print(f"Generating {n} random numbers.")
        list_to_sort = [random.randint(0, 99) for _ in range(n)]
        print("Random numbers generated:", list_to_sort)

    list_sorted = bubble_sort(list_to_sort)

    print("Sorted numbers:", list_sorted)

if __name__ == "__main__":
    main()
