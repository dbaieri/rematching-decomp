/**
 * @file        voronoifps.cpp
 * 
 * @brief       Implementation of rmt::VoronoiPartitioning.
 * 
 * @author      Filippo Maggioli\n
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date        2024-01-15
 */
#include <rmt/voronoifps.hpp>
#include <random>
#include <queue>
#include <iostream>

rmt::VoronoiPartitioning::VoronoiPartitioning(const rmt::Mesh& M)
    : VoronoiPartitioning(M, 0) {}

rmt::VoronoiPartitioning::VoronoiPartitioning(const rmt::Mesh& M, bool empty)
    : VoronoiPartitioning(M, empty, 0) {}

rmt::VoronoiPartitioning::VoronoiPartitioning(const rmt::Mesh& M, int seed)
    : VoronoiPartitioning(M, false, seed) {}

rmt::VoronoiPartitioning::VoronoiPartitioning(const rmt::Mesh& M, bool empty, int seed)
    : m_G(M.GetVertices(), M.GetTriangles()), isEmpty(empty)
{
    m_Partitions.setConstant(M.NumVertices(), 0);
    if (isEmpty) {
        return;
    }
    std::mt19937 Eng(seed);
    std::uniform_int_distribution<int> Distr(0, M.NumVertices() - 1);
    int FirstSample = Distr(Eng);
    m_Samples.emplace_back(FirstSample);
    m_G.DijkstraDistance(FirstSample, m_Distances);
    m_HDists = new cut::MinHeap(m_Distances.data(), M.NumVertices(), true);
}

rmt::VoronoiPartitioning::VoronoiPartitioning(rmt::VoronoiPartitioning&& VP)
    : m_G(std::move(VP.m_G)), isEmpty(false)
{
    m_Samples = std::move(VP.m_Samples);
    m_Partitions = std::move(VP.m_Partitions);
    m_Distances = std::move(VP.m_Distances);
    m_HDists = VP.m_HDists;
    VP.m_HDists = nullptr;
}

rmt::VoronoiPartitioning& rmt::VoronoiPartitioning::operator=(rmt::VoronoiPartitioning&& VP)
{
    m_G = std::move(VP.m_G);
    m_Samples = std::move(VP.m_Samples);
    m_Partitions = std::move(VP.m_Partitions);
    m_Distances = std::move(VP.m_Distances);
    m_HDists = VP.m_HDists;
    VP.m_HDists = nullptr;
    isEmpty = false;

    return *this;
}

rmt::VoronoiPartitioning::~VoronoiPartitioning()
{
    if (m_HDists != nullptr)
        delete m_HDists;
}



double rmt::VoronoiPartitioning::GetDistance(int i) const
{
    CUTCheckGEQ(i, 0);
    CUTCheckLess(i, m_G.NumVertices());
    return m_Distances[i];
}

const Eigen::VectorXd& rmt::VoronoiPartitioning::GetDistances() const
{
    return m_Distances;
}

int rmt::VoronoiPartitioning::GetPartition(int i) const
{
    CUTCheckGEQ(i, 0);
    CUTCheckLess(i, m_G.NumVertices());
    return m_Partitions[i];
}

const Eigen::VectorXi& rmt::VoronoiPartitioning::GetPartitions() const
{
    return m_Partitions;
}


int rmt::VoronoiPartitioning::NumSamples() const { return m_Samples.size(); }
int rmt::VoronoiPartitioning::GetSample(int i) const
{
    CUTCheckGEQ(i, 0);
    CUTCheckLess(i, NumSamples());
    return m_Samples[i];
}
const std::vector<int>& rmt::VoronoiPartitioning::GetSamples() const
{
    return m_Samples;
}


int rmt::VoronoiPartitioning::FarthestVertex() const
{
    return (int)m_HDists->FindMin().second;
}

void rmt::VoronoiPartitioning::AddSample(int NewSample)
{
    if (isEmpty) {
        m_Samples.emplace_back(NewSample);
        m_G.DijkstraDistance(NewSample, m_Distances);
        m_HDists = new cut::MinHeap(m_Distances.data(), m_Partitions.size(), true);
        isEmpty = false;
        return;
    }
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>,
                        std::greater<std::pair<double, int>>> Q;
    m_Distances[NewSample] = 0;
    m_Partitions[NewSample] = NumSamples();
    Q.emplace(0, NewSample);
    while (!Q.empty())
    {
        std::pair<double, int> Next = Q.top();
        Q.pop();

        int Cur = Next.second;
        double W = Next.first;

        if (W < m_HDists->GetKey(Cur))
            m_HDists->SetKey(Cur, W);

        int Deg = m_G.NumAdjacents(Cur);
        for (int j = 0; j < Deg; ++j)
        {
            WEdge Neig = m_G.GetAdjacent(Cur, j);
            if (m_Distances[Neig.first] <= W + Neig.second)
                continue;
            m_Distances[Neig.first] = W + Neig.second;
            Q.emplace(m_Distances[Neig.first], Neig.first);
            m_Partitions[Neig.first] = NumSamples();
        }
    }

    m_Samples.emplace_back(NewSample);
}



rmt::VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(const rmt::Mesh& M) 
    : VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(M, 0) {}

rmt::VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(const rmt::Mesh& M, bool empty) 
    : VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(M, empty, 0) {}

rmt::VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(const rmt::Mesh& M, int seed) 
    : VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(M, false, seed) {}

rmt::VoronoiPartitioningExplicit::VoronoiPartitioningExplicit(const rmt::Mesh& M, bool empty, int seed)
    : VoronoiPartitioning::VoronoiPartitioning(M, isEmpty, seed)
{
    if (!isEmpty) {
        ExplicitRegions.push_back({});
        for (int i = 0; i < m_Partitions.size(); i++) 
            ExplicitRegions[0].insert(i); 
    }
}

void rmt::VoronoiPartitioningExplicit::AddSample(int NewSample) 
{
    if (isEmpty) {
        m_Samples.emplace_back(NewSample);
        m_G.DijkstraDistance(NewSample, m_Distances);
        m_HDists = new cut::MinHeap(m_Distances.data(), m_Partitions.size(), true);
        ExplicitRegions.push_back({});
        for (int i = 0; i < m_Partitions.size(); i++) 
            ExplicitRegions[0].insert(i); 
        isEmpty = false;
        return;
    }
    std::priority_queue<std::pair<double, int>,
                        std::vector<std::pair<double, int>>,
                        std::greater<std::pair<double, int>>> Q;
    m_Distances[NewSample] = 0;
    m_Partitions[NewSample] = NumSamples();
    ExplicitRegions.push_back({});

    Q.emplace(0, NewSample);
    while (!Q.empty())
    {
        std::pair<double, int> Next = Q.top();
        Q.pop();

        int Cur = Next.second;
        double W = Next.first;

        if (W < m_HDists->GetKey(Cur))
            m_HDists->SetKey(Cur, W);

        int Deg = m_G.NumAdjacents(Cur);
        for (int j = 0; j < Deg; ++j)
        {
            WEdge Neig = m_G.GetAdjacent(Cur, j);
            if (m_Distances[Neig.first] <= W + Neig.second)
                continue;
            m_Distances[Neig.first] = W + Neig.second;
            Q.emplace(m_Distances[Neig.first], Neig.first);
            auto OldPart = m_Partitions[Neig.first];
            m_Partitions[Neig.first] = NumSamples();
            ExplicitRegions[OldPart].erase(Neig.first);
            ExplicitRegions[NumSamples()].insert(Neig.first);
        }
    }

    m_Samples.emplace_back(NewSample);
}


const std::vector<std::unordered_set<int>>& rmt::VoronoiPartitioningExplicit::GetRegions() const 
{
    return ExplicitRegions;
}

rmt::VoronoiPartitioningExplicit::~VoronoiPartitioningExplicit()
{
    // VoronoiPartitioning::~VoronoiPartitioning();
}