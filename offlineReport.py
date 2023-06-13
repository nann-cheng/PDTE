def get_total_value(lineNumber):
    totoal=0

    for i in range(3):
        file_path = "offline-player"+str(i)+".txt"
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                totoal += int(lines[lineNumber-1].strip())  # array indices start at 0, so the third line is at index 2
        except IndexError:
            print(f"The file doesn't have at least three lines.")
        except ValueError:
            print(f"The third line in the file is not an integer.")
        except Exception as e:
            print(f"An error occurred: {e}")
    # print(prefix,totoal/1024," bytes in total." )
    return totoal


featureOffline = get_total_value(7)/1024
# featureOnline = get_total_value(13)/1024


print(f" ${featureOffline:.2f}$ ")



# cmpOffline = get_total_value(11)/1024
# cmpOnline = get_total_value(16)/1024

# evalOffline = get_total_value(9)/1024
# evalOnline = get_total_value(20)/1024

# allOnline = featureOnline+cmpOnline+evalOnline

# allOffline = featureOffline+cmpOffline+evalOffline

# print(f"${featureOnline:.2f}$  &  ${featureOffline:.2f}$ & ${cmpOnline:.2f}$ & ${cmpOffline:.2f}$ & ${evalOnline:.2f}$ & ${evalOffline:.2f}$ & ${allOnline:.2f}$ & ${allOffline:.2f}$")