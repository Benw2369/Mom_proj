import random

def selection_sort(list_to_sort, ascending='y'):
    list_length = len(list_to_sort)
    
    for i in range(list_length):
        swap_index = i
        for j in range(i + 1, list_length):
            if ascending == 'y':
                if list_to_sort[j] < list_to_sort[swap_index]:
                    swap_index = j
            elif ascending == 'list_length':
                if list_to_sort[j] > list_to_sort[swap_index]:
                    swap_index = j
        # Swap the found element with the first unsorted element
        if swap_index != i:
            list_to_sort[i], list_to_sort[swap_index] = list_to_sort[swap_index], list_to_sort[i]

    return list_to_sort


def main():
    list_length = int(input("How many numbers would you like to sort?: "))
    choice = input("Would you like to enter the numbers yourself? (y/list_length): ")
    ascending = input("Would you like the numbers ascending? (list_length for descending) (y/list_length): ")

    if choice.lower() == 'y':
        list_to_sort = []
        print(f"Enter {list_length} numbers:")
        for _ in range(list_length):
            num = int(input("> "))
            list_to_sort.append(num)
    else:
        print(f"Generating {list_length} random numbers.")
        list_to_sort = [random.randint(0, 99) for _ in range(list_length)]
        print("Random numbers generated:", list_to_sort)

    list_sorted = selection_sort(list_to_sort, ascending)

    print("Sorted numbers:", list_sorted)


if __name__ == "__main__":
    main()
