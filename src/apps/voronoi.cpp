/**
 * @file        test_app.cpp
 * 
 * @brief       Sample application for testing features.
 * 
 * @author      Filippo Maggioli\n
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date        2023-07-17
 */
#define NOMINMAX
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <rmt/rmt.hpp>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <unsupported/Eigen/SparseExtra>
#include <nlohmann/json.hpp>

#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/doublearea.h>
// namespace Eigen
// {
//     Eigen::internal::all_t all = Eigen::placeholders::all;
// };
// #include <igl/remove_unreferenced.h>
// #include <igl/remove_duplicate_vertices.h>

#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>

void StartTimer();
double StopTimer();

struct rmtArgs
{
    std::string DataDir;
    std::string OutDir;
    int RemeshCount;
    int Seed;
};

rmtArgs ParseArgs(int argc, const char* const argv[]);
void Usage(const std::string& Prog, bool IsError = false);

int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config);

template<typename T>
T RandomSampleSet(std::unordered_set<T>& s, std::mt19937& gen) {
    std::uniform_int_distribution<size_t> bucket_dist(0, s.bucket_count() - 1);

    while (true) {
        size_t bucket = bucket_dist(gen);

        if (s.bucket_size(bucket) == 0)
            continue;

        std::uniform_int_distribution<size_t> elem_dist(0, s.bucket_size(bucket) - 1);

        auto it = s.begin(bucket);
        std::advance(it, elem_dist(gen));
        return *it;
    }
}

int main(int argc, const char* const argv[])
{
    auto Args = ParseArgs(argc, argv);
    double TotTime = 0.0;
    double t;

    std::filesystem::path OutDir(Args.OutDir);
    if (!std::filesystem::exists(OutDir)) {
        std::filesystem::create_directories(OutDir);
    }

    std::filesystem::path DataDir(Args.DataDir);
    int Return;
    for (auto const& Content : std::filesystem::directory_iterator{DataDir}) {

        if (!std::filesystem::is_regular_file(Content)) continue;

        t = 0.0f;
        StartTimer();
        Return = ProcessMesh(Content, Args);
        if (Return == -1) {        // Breaking error
            return 1;
        } else if (Return == -2) { // Skip shape: Flat Union refinement failed
            continue;
        } else {
            t = StopTimer();
            TotTime += t;
        }

    }

    std::cout << "Program terminated successfully in " << TotTime << "s." << std::endl;
    return 0;
}

int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config) {

    std::string FileStr = File.string();
    std::cout << "Processing mesh " << FileStr << "... " << std::endl;
    rmt::Mesh Mesh(FileStr);

    size_t NVOrig = Mesh.NumVertices();
    size_t NTOrig = Mesh.NumTriangles();

    std::cout << "Repairing non manifoldness... " << std::endl;
    Mesh.MakeManifold();
    size_t NVManif = Mesh.NumVertices();
    size_t NTManif = Mesh.NumTriangles();

    std::cout << "Normalizing mesh... " << std::endl;
    Eigen::VectorXd FaceAreas;
    igl::doublearea(Mesh.GetVertices(), Mesh.GetTriangles(), FaceAreas);
    Mesh.CenterAtOrigin();
    double Area = (FaceAreas.array() / 2).sum();
    Mesh.Scale(1.0 / std::sqrt(Area));
    
    std::cout << "Computing mesh edges and boundaries... " << std::endl;
    Mesh.ComputeEdgesAndBoundaries();


    std::cout << "Computing decompositions with count " << Config.RemeshCount << "..." << std::endl;
    size_t PerIterationSamples = Config.RemeshCount;
    std::mt19937 Eng(Config.Seed);

    std::vector<std::vector<unsigned int>> PerVertexPartitions(Mesh.NumVertices());
    // std::unordered_set<unsigned int> UnsampledSet;
    std::vector<bool> Sampled(Mesh.NumVertices(), false);
    std::vector<int> Permutation(Mesh.NumVertices());
    for (int i = 0; i < Mesh.NumVertices(); i++) {
        //UnsampledSet.insert(i);
        Permutation.push_back(i);
    }
    // std::shuffle(Permutation.begin(), Permutation.end(), Eng);

    int Loops = 0, Index = 0, SampledCount = 0, FirstSample = 0;
    std::cout << "Begin sampling:" << std::endl;
    while (SampledCount < Mesh.NumVertices()) {
  
        // FirstSample = RandomSampleSet<unsigned int>(UnsampledSet, Eng); // = Distr(Eng);
        // std::sample(UnsampledSet.begin(), UnsampledSet.end(), &FirstSample, 1, Eng);
        
        if (Index == 0) {
            FirstSample = 0;
            Index = 1;
        } else {
            for (int i = Index; i < Mesh.NumVertices(); i++) {
                if (!Sampled[i]) {
                    FirstSample = i;
                    Index = i+1;
                    break;
                }
            }
        }

        //std::cout << "\tSampling mesh with FPS and computing regions..." << std::endl;
        rmt::VoronoiPartitioningExplicit VPart(Mesh, true);
        VPart.AddSample(FirstSample);
        int Sample;
        std::vector<int> NewSamples;
        while (VPart.NumSamples() < PerIterationSamples) {
            Sample = VPart.FarthestVertex();
            if (!Sampled[Sample]) 
                NewSamples.push_back(Sample);
            VPart.AddSample(Sample);
        }
        

        //std::cout << "\tStoring regions for current sampling..." << std::endl;
        /* Compute cluster index for each high-res vertex */
        /*
        Eigen::VectorXi PartVec = VPart.GetPartitions();
        for (int i = 0; i < PartVec.size(); i++) {  // All mesh vertices
            if (!Sampled[PartVec[i]]) 
                PerVertexPartitions[PartVec[i]].push_back(i);
        }
        */
        auto Regions = VPart.GetRegions();
        int i = 0;
        for (auto Center = NewSamples.begin(); Center != NewSamples.end(); Center++) {
            PerVertexPartitions[*Center].reserve(Regions[i].size());
            std::copy(Regions[i].begin(), Regions[i].end(), PerVertexPartitions[*Center].begin());
            Sampled[*Center] = true;
            SampledCount++;
            i++;
        }
        
        std::cout << "\r\tSampled " << ((float)SampledCount / (float)Mesh.NumVertices()) * 100.f << "% of the shape..." << std::flush;
        Loops++;

        // std::cout << (SampledCount) << "<" << Mesh.NumVertices() << "?" << std::endl;

    }

    std::cout << "Completed in " << Loops << " iterations." << std::endl;

    std::cout << std::endl << "Exporting... " << std::endl;
    std::string FileName = File.stem().string();
    std::filesystem::path OutDir(Config.OutDir);
    std::filesystem::path VoronoiOutFile = OutDir / (FileName + "_regions.ply");
    if (!rmt::ExportVoronoi(VoronoiOutFile.string(), PerVertexPartitions))
    {
        std::cerr << "Cannot write output." << std::endl;
        return -1;
    }

    return 0;
}


std::chrono::system_clock::time_point Start;
void StartTimer()
{
    Start = std::chrono::system_clock::now();
}

double StopTimer()
{
    std::chrono::system_clock::time_point End;
    End = std::chrono::system_clock::now();
    std::chrono::system_clock::duration ETA;
    ETA = End - Start;
    size_t ms;
    ms = std::chrono::duration_cast<std::chrono::milliseconds>(ETA).count();
    return ms * 1.0e-3;
}

rmtArgs ParseArgs(int argc, const char* const argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        if (argvi == "-h" || argvi == "--help")
        {
            Usage(argv[0]);
            exit(0);
        }
    }

    rmtArgs Args;
    Args.DataDir = "";
    Args.OutDir = "";
    Args.RemeshCount = 1000;
    Args.Seed = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        if (argvi == "-c" || argvi == "--count")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.RemeshCount = std::stoi(argv[++i]);
            continue;
        }
        if (argvi == "-o" || argvi == "--output")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.OutDir = argv[++i];
            continue;
        }
        if (argvi == "-s" || argvi == "--seed")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.Seed = std::stoi(argv[++i]);
            continue;
        }
        if (Args.DataDir.empty()) Args.DataDir = argvi;
    }

    if (Args.DataDir.empty())
    {
        std::cerr << "No data directory given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.OutDir.empty())
    {
        std::filesystem::path OutDir = std::filesystem::path(Args.DataDir) / "decomp";
        Args.OutDir = OutDir.string();
    }

    return Args;
}

void Usage(const std::string& Prog, bool IsError)
{
    std::ostream* _out = &std::cout;
    if (IsError)
        _out = &std::cerr;
    std::ostream& out = *_out;

    out << std::endl;
    out << Prog << " usage:" << std::endl;
    out << std::endl;
    out << "\t" << Prog << " data_dir [-o|--output out_dir] [-s|--seed seed] [-c|--count samples_per_iter]" << std::endl;
    out << "\t" << Prog << " -h|--help" << std::endl;
    out << std::endl;
    out << "Arguments details:" << std::endl;
    out << "\t- data_dir is the directory containing the mesh dataset;" << std::endl;
    out << "\t- -c|--count sets the sampling size per iteration;" << std::endl;
    out << "\t- -s|--seed sets the seed for random generation (used for selecting remesh vertices);" << std::endl;
    out << "\t- -o|--output sets the output directory for the processed dataset;" << std::endl;
    out << "\t- -f|--file sets the arguments using the content of config_file." << std::endl;
    out << "\t- -h|--help prints this message." << std::endl;

    if (IsError)
        exit(-1);
}

