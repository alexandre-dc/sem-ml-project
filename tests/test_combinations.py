import time

arr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29, 31]
# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

index = 0
pointer = 0
start = 0
end = len(arr)

def combinationUtil(arr, data, start,  
                    end, index, r, all_combinations): 

    if (index == r): 
        data_temp = []
        for j in range(r): 
            data_temp.append(data[j])
        all_combinations.append(data_temp)
        return

    i = start;  
    while(i < end and end - i + 1 >= r - index): 
        data[index] = arr[i]; 
        combinationUtil(arr, data, i + 1,  
                        end, index + 1, r, all_combinations); 
        i += 1; 

#combinationUtil(arr, data, start, end, index, r, all_combinations)

# t0 = time.clock()
# for i in range(2, 3):
#     all_combinations = []
#     r = i
#     data = [0] * r
#     combinationUtil(arr, data, start, end, index, r, all_combinations)
#     for comb in all_combinations:
#         print(sum(comb), comb)
        #print(total)

for i in range(2, 12):
    all_combinations = []
    r = i
    data = [0] * r
    combinationUtil(arr, data, start, end, index, r, all_combinations)
    canonic_scores = []
    for comb in all_combinations:
        total = 1
        for j in comb:
            total *= j
        # total = 0
        # for j in comb:
        #     total += 10 ** (12 - j)
        canonic_scores.append(total)
        print(total)
    
#     for x in range(len(canonic_scores)):
#         for y in range(x + 1, len(canonic_scores)):
#             if canonic_scores[x] == canonic_scores[y]:
#                 print("error")


print(time.clock())


