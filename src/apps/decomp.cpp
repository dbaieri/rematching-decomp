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
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cmath>

void StartTimer();
double StopTimer();

struct rmtArgs
{
    std::string InFile;
    std::string OutDir;
    float RemeshPctg;
    bool UsePctg;
    int RemeshCount;
    bool UseCount;
    int RNG;
    bool Visualize;
    unsigned int MaxIter;
};

rmtArgs ParseArgs(int argc, const char* const argv[]);
void Usage(const std::string& Prog, bool IsError = false);

int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config);

int main(int argc, const char* const argv[])
{
    auto Args = ParseArgs(argc, argv);
    double TotTime = 0.0;
    double t;

    if (Args.Visualize) {
        polyscope::init();
    }

    std::filesystem::path OutDir(Args.OutDir);
    if (!std::filesystem::exists(OutDir)) {
        std::filesystem::create_directories(OutDir);
    }

    std::filesystem::path InPath(Args.InFile);
    t = 0.0f;
    StartTimer();
    int Return = ProcessMesh(InPath, Args);
    t = StopTimer();

    std::cout << "Program terminated successfully in " << t << "s." << std::endl;
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

    std::cout << "Computing Voronoi FPS with density " << Config.RemeshPctg << "..." << std::endl;
    size_t NVRemesh;
    if (Config.UsePctg) {
        NVRemesh = std::floor(Config.RemeshPctg * NVOrig);
    } else if (Config.UseCount) {
        NVRemesh = Config.RemeshCount;
    }
    rmt::VoronoiPartitioning VPart(Mesh, Config.RNG);
    while (VPart.NumSamples() < NVRemesh)
        VPart.AddSample(VPart.FarthestVertex());

    std::cout << "Refining sampling to ensure closed ball property... " << std::endl;
    rmt::FlatUnion FU(Mesh, VPart);
    unsigned int NumIters = 0;
    bool RefinementError = false;
    do
    {
        if (Config.MaxIter > 0 && NumIters >= Config.MaxIter) {
            RefinementError = true;
            break;
        }
        FU.DetermineRegions();
        FU.ComputeTopologies();
        NumIters++;
    } while (!FU.FixIssues());
    if (RefinementError) {
        std::cerr << "Cannot ensure closed ball property on this shape! Skipping" << std::endl;
        return -2;
    }
    size_t NVRefined = VPart.NumSamples();
    
    /* Compute cluster index for each high-res vertex */
    Eigen::VectorXd PartVec = VPart.GetPartitions().cast<double>();
    Eigen::VectorXd Centers = Eigen::VectorXd::Zero(PartVec.rows());
    auto VSamples = VPart.GetSamples();
    for (auto ptr = VSamples.begin(); ptr != VSamples.end(); ptr++) {
        Centers(*ptr) = 1.0;
    }
    Eigen::MatrixXd FeatVec(PartVec.rows(), 2);
    FeatVec << PartVec, Centers;
    
    if (Config.Visualize) {
        polyscope::registerSurfaceMesh("Original Mesh", Mesh.GetVertices(), Mesh.GetTriangles());
        polyscope::getSurfaceMesh("Original Mesh")->addVertexScalarQuantity("Voronoi Regions", PartVec.cast<double>());
        polyscope::show();
    }

    std::cout << "Reconstructing mesh... " << std::endl;
    Eigen::MatrixXd VV;
    Eigen::MatrixXi FF;
    rmt::MeshFromVoronoi(Mesh.GetVertices(), Mesh.GetTriangles(), VPart, VV, FF);
    // rmt::CleanUp(VV, FF);
    size_t NVFinal = VV.rows();
    size_t NTFinal = FF.rows();

    std::cout << "== Current mesh statistics ==" << std::endl;
    std::cout << "\t(Original mesh) Vertex count: " << NVOrig << " -- Triangle count: " << NTOrig << std::endl;
    std::cout << "\t(Manifold repair) Vertex count: " << NVManif << " -- Triangle count: " << NTManif << std::endl;
    std::cout << "\t(Voronoi sampling) Base sample size: " << NVRemesh << " -- Refined sample: " << NVRefined << std::endl;
    std::cout << "\t(Remeshing) Vertex count: " << NVFinal << " -- Triangle count: " << NTFinal << std::endl;
    if (FF.rows() == 0)
    {
        std::cerr << "WARNING: Remesh has zero triangles. Maybe there are too many connected components?" << std::endl;
    }

    std::cout << "Exporting... " << std::endl;
    std::string FileName = File.stem().string();
    std::filesystem::path OutDir(Config.OutDir);
    std::filesystem::path RemeshOutFile = OutDir / (FileName + "_remesh.ply");
    std::filesystem::path VoronoiOutFile = OutDir / (FileName + "_voronoi.ply");
    if (!rmt::ExportMesh(RemeshOutFile.string(), VV, FF))
    {
        std::cerr << "Cannot write mesh." << std::endl;
        return -1;
    }
    if (!rmt::ExportMesh(VoronoiOutFile.string(), Mesh.GetVertices(), Mesh.GetTriangles(), FeatVec))
    {
        std::cerr << "Cannot write mesh." << std::endl;
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
    Args.InFile = "";
    Args.OutDir = "";
    Args.RemeshPctg = 1.0;
    Args.UsePctg = false;
    Args.RemeshCount = 0;
    Args.UseCount = false;
    Args.Visualize = false;
    Args.MaxIter = 0;
    Args.RNG = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        if (argvi == "-p" || argvi == "--pctg")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.RemeshPctg = std::stof(argv[++i]);
            Args.UsePctg = true;
            continue;
        }
        if (argvi == "-c" || argvi == "--count")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.RemeshCount = std::stoi(argv[++i]);
            Args.UseCount = true;
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
            Args.RNG = std::stoi(argv[++i]);
            continue;
        }
        if (argvi == "-m" || argvi == "--max_iter")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.MaxIter = std::stoi(argv[++i]);
            continue;
        }
        if (argvi == "-v" || argvi == "--visual")
        {
            Args.Visualize = true;
            continue;
        }
        if (Args.InFile.empty()) Args.InFile = argvi;
    }

    if (Args.UsePctg == Args.UseCount) {
        std::cerr << "You must specify exactly one between vertex percentage and vertex count." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.InFile.empty())
    {
        std::cerr << "No input file given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.RemeshPctg == -1)
    {
        std::cerr << "No remeshing density percentage given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.OutDir.empty())
    {
        auto InPath = std::filesystem::path(Args.InFile);
        Args.OutDir = InPath.parent_path().string();
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
    out << "\t" << Prog << " in_file [-p|--pctg remesh_pctg] [-c|--count remesh_count] [-o|--output out_dir] [-s|--seed rng] [-v|--visual] [-m|--max_iter max_refinement_steps]" << std::endl;
    out << "\t" << Prog << " -h|--help" << std::endl;
    out << std::endl;
    out << "Arguments details:" << std::endl;
    out << "\t- in_file is the input mesh file;" << std::endl;
    out << "\t- -p|--pctg sets the fraction of input vertices you want in the remesh (one between this and -c must be specified);" << std::endl;
    out << "\t- -c|--count explicitly sets the vertex count of the remesh (one between this and -p must be specified);" << std::endl;
    out << "\t- -o|--output sets the directory for program output;" << std::endl;
    out << "\t- -s|--seed sets the seed for random generation (used for selecting remesh vertices);" << std::endl;
    out << "\t- -v|--visual if provided, the script will show Voronoi decompositions via Polyscope as it runs;" << std::endl;
    out << "\t- -f|--file sets the arguments using the content of config_file." << std::endl;
    out << "\t- -m|--max_iter sets the maximum number of iterations for flat union refinement before failure." << std::endl;
    out << "\t- -h|--help prints this message." << std::endl;

    if (IsError)
        exit(-1);
}

