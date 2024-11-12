# Ex.No: 2  Implementation of Depth First Search
### DATE:                                                                            
### REGISTER NUMBER : 212222220052
### AIM: 
To write a python program to implement Depth first Search.

### Algorithm:
1. Start the program
2. Create the graph by using adjacency list representation
3. Define a function dfs and take the set “visited” is empty 
4. Search start with initial node. Check the node is not visited then print the node.
5. For each neighbor node, recursively invoke the dfs search.
6. Call the dfs function by passing arguments visited, graph and starting node.
7. Stop the program.
   
### Program:
```
graph = {
 'A' : ['B','C'],
 'B' : ['D', 'E'],
 'C' : ['F','G'],
 'D' : [],
 'E' : [],
 'F' : [],
 'G' : []
 }
visited = set() # Set to keep track of visited nodes of graph.
def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
# Driver Code
print("Following is the Depth-First Search")
dfs(visited, graph, 'A')
```

### Output:
![image](https://github.com/user-attachments/assets/5b3ba570-5626-4567-846f-aa33c9a07c25)

### Result:
Thus the depth first search order was found sucessfully.
