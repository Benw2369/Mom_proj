import random

def merge_sort(list_to_sort, ascending='y'):
    if len(list_to_sort) <= 1:
        return list_to_sort

    mid = len(list_to_sort) // 2
    left_half = merge_sort(list_to_sort[:mid], ascending)
    right_half = merge_sort(list_to_sort[mid:], ascending)

    return merge(left_half, right_half, ascending)

def merge(left, right, ascending):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if ascending == 'y':
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        elif ascending == 'list_length':  # descending
            if left[i] >= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

    # Append any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


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

    list_sorted = merge_sort(list_to_sort, ascending)

    print("Sorted numbers:", list_sorted)


if __name__ == "__main__":
    main()
