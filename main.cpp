// C++17. Generates graphs and requests automatically; settings below.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace std;
using ll = long long;

// Change these settings to run different experiments. No console input is needed.
const vector<pair<int, int>> NETWORK_SIZES = {
    {10, 15}, {50, 100}, {100, 250}, {500, 1500}
}; // Each pair is {vertices, roads}.
constexpr unsigned int RANDOM_SEED = 2307003;
constexpr int TIME_STEPS = 3;
constexpr int MIN_DISTANCE = 1, MAX_DISTANCE = 20;
constexpr int MIN_REQUESTS = 5, MAX_REQUESTS = 10;
constexpr int MIN_VEHICLES = 1, MAX_VEHICLES = 20;
constexpr double LAMBDA = 1.0;
constexpr double LOAD_POWER = 2.0;


struct Edge { int u, v, distance; };
struct Request { int vehicles, source, destination; };
enum Method { SHORTEST, GREEDY, BALANCED };

struct Graph {
    vector<Edge> edges;
    vector<vector<pair<int, int>>> adj;
    explicit Graph(int n) : adj(n) {}
    void add(int u, int v, int distance) {
        int id = static_cast<int>(edges.size());
        edges.push_back({u, v, distance});
        adj[u].push_back({v, id});
        adj[v].push_back({u, id});
    }
};


vector<int> findPath(const Graph& g, int source, int destination,
                     const vector<ll>& load, Method method, double lambda, double power) {
    int n = static_cast<int>(g.adj.size());
    vector<double> dist(n, numeric_limits<double>::infinity());
    vector<int> parent(n, -1), parentEdge(n, -1);
    using State = pair<double, int>;
    priority_queue<State, vector<State>, greater<State>> pq;
    dist[source] = 0;
    pq.push({0, source});
    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du > dist[u]) continue;
        if (u == destination) break;
        for (auto [v, id] : g.adj[u]) {
            // Increase in L^power when one vehicle is added.
            double marginal = 0.0;
            if (method == GREEDY || (method == BALANCED && lambda > 0)) {
                double L = static_cast<double>(load[id]);
                if (power == 1.0) marginal = 1.0;
                else if (power == 2.0) marginal = 2.0 * L + 1.0;
                else if (L == 0.0) marginal = 1.0;
                else {
                    // Stable form of (L + 1)^power - L^power.
                    marginal = pow(L, power) * expm1(power * log1p(1.0 / L));
                }
                if (!isfinite(marginal) || marginal <= 0)
                    throw runtime_error("Load penalty exceeds numerical range; reduce power");
            }
            double cost = g.edges[id].distance;
            if (method == GREEDY) cost = marginal;
            if (method == BALANCED) cost += lambda * marginal;
            if (!isfinite(cost) || !isfinite(du + cost))
                throw runtime_error("Routing cost overflow; reduce lambda or power");
            if (du + cost < dist[v]) {
                dist[v] = du + cost;
                parent[v] = u;
                parentEdge[v] = id;
                pq.push({dist[v], v});
            }
        }
    }
    if (!isfinite(dist[destination])) throw runtime_error("Graph is disconnected");
    vector<int> path;
    for (int v = destination; v != source; v = parent[v])
        path.push_back(parentEdge[v]);
    reverse(path.begin(), path.end());
    return path;
}

void solve(const Graph& g, const vector<Request>& requests,
           Method method, double lambda, double power) {
    const string names[] = {"Shortest-path routing", "Greedy load-balancing routing",
                            "Balanced routing"};
    int m = static_cast<int>(g.edges.size());
    vector<ll> load(m, 0); // Reset for every method and time step.
    vector<map<vector<int>, ll>> assignments(requests.size());
    auto start = chrono::steady_clock::now();

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& r = requests[i];
        int remaining = r.vehicles;
        while (remaining > 0) {
            auto path = findPath(g, r.source, r.destination, load, method, lambda, power);
            int count = method == SHORTEST ? remaining : 1;
            vector<int> vertices = {r.source};
            int u = r.source;
            for (int id : path) {
                load[id] += count;
                const auto& e = g.edges[id];
                u = e.u == u ? e.v : e.u;
                vertices.push_back(u);
            }
            assignments[i][vertices] += count;
            remaining -= count;
        }
    }

    ll maximum = 0, totalDistance = 0;
    double average = 0, variance = 0;
    for (int i = 0; i < m; ++i) {
        maximum = max(maximum, load[i]);
        average += load[i];
        totalDistance += load[i] * g.edges[i].distance;
    }
    average /= m;
    for (ll x : load) variance += (x - average) * (x - average);
    variance /= m;
    double ms = chrono::duration<double, milli>(
        chrono::steady_clock::now() - start).count();

    cout << "\n" << names[method] << '\n';
    for (size_t i = 0; i < requests.size(); ++i) {
        cout << "Request " << i + 1 << ":\n";
        for (const auto& [path, count] : assignments[i]) {
            cout << "  ";
            for (size_t j = 0; j < path.size(); ++j)
                cout << (j ? " -> " : "") << path[j];
            cout << " : " << count << " vehicle(s)\n";
        }
    }
    cout << "Road loads:\n";
    for (int i = 0; i < m; ++i)
        cout << "  " << g.edges[i].u << " - " << g.edges[i].v
             << " : " << load[i] << '\n';
    cout << "Maximum load: " << maximum << '\n'
         << "Average load: " << average << '\n'
         << "Load variance: " << variance << '\n'
         << "Total travel distance: " << totalDistance << '\n';
    // Timing goes to stderr so stdout is reproducible for testing.
    cerr << names[method] << " runtime: " << ms << " ms\n";
}

// A random spanning tree guarantees connectivity. Additional unique roads
// introduce alternative routes. This does not sample all connected graphs uniformly.
Graph generateGraph(int n, int m, mt19937& rng) {
    if (n < 2 || m < n - 1 || m > 1LL * n * (n - 1) / 2)
        throw runtime_error("Invalid generated graph size");
    Graph g(n);
    set<pair<int, int>> used;
    uniform_int_distribution<int> distance(MIN_DISTANCE, MAX_DISTANCE);
    vector<int> vertices(n);
    iota(vertices.begin(), vertices.end(), 0);
    shuffle(vertices.begin(), vertices.end(), rng);

    auto addRoad = [&](int u, int v) {
        if (u == v) return false;
        if (!used.insert({min(u, v), max(u, v)}).second) return false;
        g.add(u, v, distance(rng));
        return true;
    };

    for (int i = 1; i < n; ++i) {
        uniform_int_distribution<int> previous(0, i - 1);
        addRoad(vertices[i], vertices[previous(rng)]);
    }

    uniform_int_distribution<int> vertex(0, n - 1);
    while (static_cast<int>(g.edges.size()) < m) {
        int u = vertex(rng);
        int v = vertex(rng);
        addRoad(u, v);
    }
    return g;
}

vector<Request> generateRequests(int n, mt19937& rng) {
    uniform_int_distribution<int> requestCount(MIN_REQUESTS, MAX_REQUESTS);
    uniform_int_distribution<int> vehicles(MIN_VEHICLES, MAX_VEHICLES);
    uniform_int_distribution<int> source(0, n - 1);
    uniform_int_distribution<int> otherVertex(0, n - 2);
    int k = requestCount(rng);
    vector<Request> requests;
    requests.reserve(k);
    for (int i = 0; i < k; ++i) {
        int s = source(rng);
        int d = otherVertex(rng);
        if (d >= s) ++d; // Uniform destination distinct from the source.
        requests.push_back({vehicles(rng), s, d});
    }
    return requests;
}

int main() {
    ios::sync_with_stdio(false);
    cout << fixed << setprecision(2);
    try {
        if (TIME_STEPS < 1 || MIN_DISTANCE < 1 || MAX_DISTANCE < MIN_DISTANCE ||
            MIN_REQUESTS < 1 || MAX_REQUESTS < MIN_REQUESTS ||
            MIN_VEHICLES < 1 || MAX_VEHICLES < MIN_VEHICLES ||
            !isfinite(LAMBDA) || LAMBDA < 0 ||
            !isfinite(LOAD_POWER) || LOAD_POWER < 1)
            throw runtime_error("Invalid experiment settings");

        mt19937 rng(RANDOM_SEED);
        cout << "Seed: " << RANDOM_SEED << "\nLambda: " << LAMBDA
             << "\nLoad power: " << LOAD_POWER << '\n';
        for (const auto& [n, m] : NETWORK_SIZES) {
            Graph g = generateGraph(n, m, rng);
            cout << "\n========================================\n"
                 << "Network: " << n << " vertices, " << m << " roads\n"
                 << "Generated roads (u v distance):\n";
            for (const auto& e : g.edges)
                cout << e.u << ' ' << e.v << ' ' << e.distance << '\n';

            for (int t = 1; t <= TIME_STEPS; ++t) {
                // Generate once, then reuse exactly the same requests for all methods.
                auto requests = generateRequests(n, rng);
                cout << "\nTime step " << t << "\nGenerated requests: "
                     << requests.size() << "\n(vehicles source destination)\n";
                for (const auto& r : requests)
                    cout << r.vehicles << ' ' << r.source << ' '
                         << r.destination << '\n';
                for (Method method : {SHORTEST, GREEDY, BALANCED})
                    solve(g, requests, method, LAMBDA, LOAD_POWER);
            }
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
