#include<bits/stdc++.h>
#include<chrono>
#include<omp.h>

using namespace std;
using namespace std::chrono;

void add_edge(vector<vector<int>>& graph, int u, int v){
    graph[v].push_back(u);
    graph[u].push_back(v);
}

void sequential_bfs(vector<vector<int>>& graph, vector<bool>& visited, int start){
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while(!q.empty()){
        int a = q.front();
        q.pop();
        cout << a << " ";

        for(int neighbour : graph[a]){
            if(!visited[neighbour]){
                q.push(neighbour);
                visited[neighbour] = true;
            }
        }
    }
    cout<<endl;
}

void sequential_dfs(vector<vector<int>>& graph, vector<bool>& visited, int node){
    visited[node] = true;
    cout << node << " ";

    for(int neighbour : graph[node]){
        if(!visited[neighbour]){
            sequential_dfs(graph, visited, neighbour);
        }
    }
}

void print_graph(vector<vector<int>>& graph){
    int n = graph.size();
    for(int i = 0; i < n; i++){
        cout << "Neighbours of " << i << " -> ";
        for(int node : graph[i]){
            cout<<node<<" ";
        }
        cout<<endl;
    }
}

//Check if node is visited, if visited then we return, if not visited we print the node and check for it's neighbours.
void parallel_dfs(vector<vector<int>>& graph, vector<bool>& visited, int node){
    bool wasvisited = true;
    #pragma omp critical
    {
        if(!visited[node]){
            wasvisited = false;
            visited[node] = true;
            cout<<node<<" visited using thread number: "<< omp_get_thread_num()<<endl;
        }
    }
    if(wasvisited)return;

    #pragma omp task
    for(int neighbour : graph[node]){
        parallel_dfs(graph, visited, neighbour);
    }
    #pragma omp taskwait
}

void parallel_bfs(vector<vector<int>>& graph, vector<bool>& visited, int start){
    queue<int> q;
    visited[start] = true;
    q.push(start);
    cout<< start << " Visited using thread number: " << omp_get_thread_num() << endl;

    while(!q.empty()){
        int levelsize = q.size();
        vector<int> currlevel;
        for(int i = 0; i < levelsize; i++){
            currlevel.push_back(q.front());
            q.pop();
        } 
        vector<int> nextlevel;
        #pragma omp parallel for
        for(int i = 0; i < levelsize; i++){     
            int node = currlevel[i];       
            for(int neighbour : graph[node]){
                #pragma omp critical
                {
                    if(!visited[neighbour]){
                        visited[neighbour] = true;
                        nextlevel.push_back(neighbour);
                        cout<< neighbour << " visited using thread number: " << omp_get_thread_num() << endl;
                    }
                }
            }
        }
        for(int node : nextlevel){
            q.push(node);
        }
    }
}

int main(){
    int v,e;
    cout<<"Enter number of vertex(v) and edges(e): ";
    cin>>v>>e;
    vector<vector<int>> graph(v);
    cout<<"Enter all edges: ";
    for(int i = 0; i < e; i++){
        int u,v;
        cin>>u>>v;
        add_edge(graph,u,v);
    }

    print_graph(graph);
    cout<<"Enter starting vertex for parallel BFS and DFS: ";
    int ver; cin>>ver;
    vector<bool> visited(v, false);

    auto start_time = high_resolution_clock::now();
    cout<<"Sequential BFS: ";
    sequential_bfs(graph, visited, ver);
    auto end_time = high_resolution_clock::now();
    auto total_duration_seq = duration_cast<milliseconds>(end_time - start_time);
    cout << "Total time for sequential bfs: " << total_duration_seq.count() << "ms\n";

    visited.assign(v, false);
    start_time = high_resolution_clock::now();
    cout<<"Parallel BFS: \n";
    parallel_bfs(graph, visited, ver);
    end_time = high_resolution_clock::now();
    auto total_duration_parallel = duration_cast<milliseconds>(end_time - start_time);
    cout<<"Total time for parallel bfs: " << total_duration_parallel.count() << "ms\n";

    visited.assign(v, false);
    start_time = high_resolution_clock::now();
    cout<<"Sequential DFS: ";
    sequential_dfs(graph, visited, ver);
    end_time = high_resolution_clock::now();
    auto total_duration_seq_dfs = duration_cast<milliseconds>(end_time - start_time);
    cout<<endl;
    cout<< "Total time for sequential dfs: " << total_duration_seq_dfs.count() << "ms\n";

    visited.assign(v, false);
    start_time = high_resolution_clock::now();
    cout<<"Parallel DFS: \n";
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_dfs(graph, visited, ver);
        }
    }  
    end_time = high_resolution_clock::now();
    auto total_duration_par_dfs = duration_cast<milliseconds>(end_time - start_time);
    cout<<endl;
    cout<< "Total time for parallel dfs: " << total_duration_par_dfs.count() << "ms\n";

    return 0;
}