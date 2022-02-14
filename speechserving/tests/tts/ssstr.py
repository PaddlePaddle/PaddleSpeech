#coding=utf-8
import time

def find_index(find_str: str, key: str) -> list:
    """find duicheng str based on a special middle index

    Args:
        find_str (str): orignal str
        index (int): middle index

    Returns:
        str: dst str
    """
    index_list = []
    count = find_str.count(key)
    if count == 0:
        return index_list
    index = 0
    for i in range(count - 1):
        index = find_str.find(key, index + 1)
        index_list.append(index)

    index_list.sort(reverse=True)  
    return index_list

def find_max(input_str: str) -> str:
    max_str = ""
    if len(input_str) == 1:
        return input_str
    for i in range(len(input_str)):
        temp_str = ""
        find_str = input_str[i:]
        if(len(find_str) < len(max_str)):
            return max_str
        key = input_str[i]
        index_list = find_index(find_str, key)
        if index_list != []:
            for index in index_list:
                left_index = 1
                right_index = index - 1
                if right_index == 0 or left_index == right_index:  # aa or aba
                    temp_str = find_str[:index+1]

                else: #example: index % 2 == 0: abcba, index % 2 == 1: abccba
                    while((index % 2 == 0 and left_index != right_index) or (index % 2 == 1 and left_index != right_index - 1)):
                        if find_str[left_index] == find_str[right_index]:
                            left_index += 1
                            right_index -= 1
                            continue
                        else:
                            break
                    if((index % 2 == 0 and left_index == right_index) or (index % 2 == 1 and left_index == right_index - 1 and find_str[left_index] == find_str[right_index])):
                        temp_str = find_str[:index+1]
                        
                if len(temp_str) > len(max_str):
                    max_str = temp_str
    return max_str

start_time = time.time()
str = "abcdabaaaaacaaaa"
print(find_max(str))
print(time.time() - start_time)
