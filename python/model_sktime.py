# Individual Assignment - Linked-List
# Done by: Sydow, Sebastian 
# Date: 2020/10/14

############ import libaries ############## 
import time
import pandas as pd 
import matplotlib.pyplot as plt

########### Create Linked List ############ 

# Create Nodes
class Node:
    # initialize the Node
    def __init__(self, data):
        # each node consists of two parts:
        # 1) data
        self.data= data  # assign data
        # 2) pointer to the next node
        self.next= None  # initialize next node as null/none
    
# Create class for the linked list which contains the nodes
class LinkedList:
    def __init__(self):
        self.head = None # initially, the linked list is empty, therefore self.head = None

    # function to insert the elements of an array as a new node at the beginning
    # the order of the array will be identical to the order of the linked list
    def create_llist (self, array):
        length_array = len(array)
        i = -1
        while i  > -length_array:
            # if the linked list is empty, new_node will be the head of the linked list
            if self.head is None:
                self.head = Node(array[-1])  # initialize head-node with last element of the array
                i -= 1  # decrement i by 1
            else:
                new_node = Node(array[i])   # allocate node and put in the data from the array
                new_node.next = self.head   # make next of new node self.head 
                self.head = new_node        # move the head to point to new node
                i -= 1  # decrement i by

    def insert (self, new_node, position):
        # create a Node out of the to be inserted element 'new_node'
        new_node = Node(new_node)

        # starting point is the Node self.head
        latest_node = self.head
        compared_position = 0    # starting position for comparions is the current position: 0
        while True:
            # compare the current position with the specified position in the function call
            if compared_position == position:
                # next node to the prevision node will be set to the new node (-> insert new node here)   
                previous_node.next = new_node
                # next node of the new node will be latest node
                new_node.next = latest_node
                break
            # if the compared positions are not equal, previous node will be set to the latest node
            previous_node = latest_node
            # latest node will be set the next node
            latest_node = latest_node.next
            # increment the compared position
            compared_position += 1 

    # function to print the linked list
    def print_llist(self):
        if self.head is None:
            print('The linked list does not contain any elements. It is empty.')
            return

        current_node = self.head
        while True:
            if current_node is None:
                break
            print(current_node.data)
            current_node = current_node.next

# # Code execution #
array = list(range(0, 1000))
llist = LinkedList()
llist.create_llist(array)
llist.insert (-1, 4)
llist.print_llist()

######### Measure Time Complexity ######### 
############### Linked List ###############
# results_ll = {}    # create empty dictionary to store the results
# element = -1    # define the element, which will be inserted to the linked list
# pos = 1         # define position at which the element will be inserted
# items = 1000 # define the starting amount of the number of elements in the linked list

# while items <= 1024000:
#     array = list(range(0, items))   # create array which will be transformed to a linked list
#     llist = LinkedList()            # create empty linked list
#     llist.create_llist(array)       # create linked list out of the given array

#     start = time.time()             # insert an element at position 1 and record the time
#     llist.insert(element, pos)
#     end = time.time()
#     total_time = end - start

#     results_ll[items] = total_time     # append the total_time to the dictionary to store the results 
#     items = items * 2               # double the number of items to reperforme the function until it exits the while-loop

# # create data frame to store results
# results_df = pd.DataFrame(list(results_ll.items()), columns = ['Number of items', 'Time for one insertion - Linked List'])

# ############### Array ###############
# results_ar = [] # create empty list to store the results
# element = -1    # define the element, which will be inserted to the linked list
# pos = 1         # define position at which the element will be inserted
# items = 1000    # define the starting amount of the number of elements in the linked list

# while items <= 1024000:
#     array = list(range(0, items))
#     start = time.time()
#     array.insert(pos, element)
#     end = time.time()
#     total_time = end - start
#     results_ar.append(total_time) 
#     items = items * 2

# # create data frame to store results
# results_df['Time for one insertion - Array'] = results_ar
# print(results_df)

# ############# Visualization ############### 
# results_df.plot(kind='line',x='Number of items',y=['Time for one insertion - Linked List', 'Time for one insertion - Array'], marker='.')
# plt.xlabel ('Number of items in linked list / array')
# plt.ylabel ('Time in seconds for one insertion')
# plt.title ('Time Complexity - Inserting one element')
# plt.xticks([1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000], 
#            ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1024k'])
# # show figure
# plt.legend()
# plt.show()