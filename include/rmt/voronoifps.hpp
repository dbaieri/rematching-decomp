/**
 * @file        voronoifps.hpp
 * 
 * @brief       Declaration of class rmt::VoronoiPartitioning.
 * 
 * @author      Filippo Maggioli\n
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date        2024-01-15
 */
#pragma once

#include <rmt/graph.hpp>
#include <rmt/mesh.hpp>
#include <cut/cut.hpp>
#include <unordered_set>


namespace rmt
{
    
class VoronoiPartitioning
{
protected:
    bool isEmpty;
    rmt::Graph m_G;
    std::vector<int> m_Samples;
    Eigen::VectorXi m_Partitions;
    Eigen::VectorXd m_Distances;
    cut::MinHeap* m_HDists;


public:
    VoronoiPartitioning(const rmt::Mesh& M);
    VoronoiPartitioning(const rmt::Mesh& M, bool empty);
    VoronoiPartitioning(const rmt::Mesh& M, int seed);
    VoronoiPartitioning(const rmt::Mesh& M, bool empty, int seed);
    VoronoiPartitioning(rmt::VoronoiPartitioning&& VP);
    rmt::VoronoiPartitioning& operator=(rmt::VoronoiPartitioning&& VP);
    ~VoronoiPartitioning();

    double GetDistance(int i) const;
    const Eigen::VectorXd& GetDistances() const;
    int GetPartition(int i) const;
    const Eigen::VectorXi& GetPartitions() const;
    
    int NumSamples() const;
    int GetSample(int i) const;
    const std::vector<int>& GetSamples() const;

    int FarthestVertex() const;
    virtual void AddSample(int NewSample);
};

class VoronoiPartitioningExplicit : public VoronoiPartitioning
{

protected:
    std::vector<std::unordered_set<int>> ExplicitRegions;

public:
    VoronoiPartitioningExplicit(const rmt::Mesh& M);
    VoronoiPartitioningExplicit(const rmt::Mesh& M, bool empty);
    VoronoiPartitioningExplicit(const rmt::Mesh& M, int seed);
    VoronoiPartitioningExplicit(const rmt::Mesh& M, bool empty, int seed);
    VoronoiPartitioningExplicit(rmt::VoronoiPartitioningExplicit&& VP);
    ~VoronoiPartitioningExplicit();

    void AddSample(int NewSample) override;
    const std::vector<std::unordered_set<int>>& GetRegions() const;
};



} // namespace rmt
