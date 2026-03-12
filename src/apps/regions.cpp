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
#include <rmt/graph.hpp>

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
    std::string InFile;
    std::string OutFile;
    float Area;
    bool Update;
};

rmtArgs ParseArgs(int argc, const char* const argv[]);
void Usage(const std::string& Prog, bool IsError = false);

int ProcessMesh(const std::filesystem::path& File, const rmtArgs& Config);


int main(int argc, const char* const argv[])
{
    auto Args = ParseArgs(argc, argv);
    double TotTime = 0.0;
    double t;

    std::filesystem::path OutDir(Args.OutFile);
    if (!std::filesystem::exists(OutDir.parent_path())) {
        std::filesystem::create_directories(OutDir.parent_path());
    }

    std::filesystem::path InFile(Args.InFile);
    t = 0.0f;
    StartTimer();
    int Return = ProcessMesh(InFile, Args);
    t = StopTimer();
    TotTime += t;

    if (Return == -1) {        // Breaking error
        return 1;
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

    std::cout << "Target area: " << Config.Area << std::endl;

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
    igl::doublearea(Mesh.GetVertices(), Mesh.GetTriangles(), FaceAreas);
    
    std::cout << "Computing mesh edges and boundaries... " << std::endl;
    Mesh.ComputeEdgesAndBoundaries();
    
    auto Vertices = Mesh.GetVertices();
    auto Faces = Mesh.GetTriangles();
    rmt::Graph m_G(Vertices, Faces);
    std::vector<std::vector<unsigned int>> PerVertexPartitions(Mesh.NumVertices());
    std::vector<std::vector<unsigned int>> FacesPerPartition(Mesh.NumVertices());

    std::vector<double> Distances(Mesh.NumVertices(), std::numeric_limits<int>::max()); 
    std::vector<bool> Visited(Mesh.NumVertices(), false);
    std::vector<int> FaceCount(Mesh.NumTriangles(), 0);
    std::vector<bool> FaceAdded(Mesh.NumTriangles(), false);
    std::vector<std::vector<int>> PerVertexFaces(Mesh.NumVertices());
    for (int j = 0; j < Mesh.NumTriangles(); j++) {
        auto Face = Faces.row(j);
        PerVertexFaces[Face[0]].push_back(j);
        PerVertexFaces[Face[1]].push_back(j);
        PerVertexFaces[Face[2]].push_back(j);
    }
    
    int Steps = 0;
    double AccumulatedArea;
    bool EarlyStop;
    for (int v0 = 0; v0 < Mesh.NumVertices(); v0++) {

        if (v0 > 0) {
            std::fill(Distances.begin(), Distances.end(), std::numeric_limits<int>::max());
            std::fill(Visited.begin(), Visited.end(), false);
            std::fill(FaceCount.begin(), FaceCount.end(), 0);
            std::fill(FaceAdded.begin(), FaceAdded.end(), false);
        }
        std::priority_queue<std::pair<double, int>,
                            std::vector<std::pair<double, int>>,
                            std::greater<std::pair<double, int>>> Queue;
        
        Distances[v0] = 0;
        Queue.emplace(0.0, v0);
        AccumulatedArea = 0.0;
        EarlyStop = false;

        // std::cout << "Vertex " << v0 << std::endl;

        while (!Queue.empty()) {

            std::pair<double, int> Next = Queue.top();
            Queue.pop();
            double W = Next.first;
            int v = Next.second;

            if (Visited[v])
                continue;
                
            // std::cout << "\t New vertex: " << v << " with gdist " << W << std::endl;

            Visited[v] = true;
            PerVertexPartitions[v0].push_back(v); // Add visited vertex to the region

            // update incident faces
            auto IncidentFaces = PerVertexFaces[v];
            int Face;
            for (auto ptr = IncidentFaces.begin(); ptr != IncidentFaces.end(); ptr++) {
                Face = *ptr;
                FaceCount[Face]++;
                if (FaceCount[Face] == 3 && !FaceAdded[Face]) {
                    // std::cout << "\t Aggregating face: " << Face << std::endl;
                    AccumulatedArea += FaceAreas[Face] / 2.0;
                    FaceAdded[Face] = true;
                    FacesPerPartition[v0].push_back(Face);
                    // std::cout << "\t Current area: " << AccumulatedArea << std::endl;
                    if ((float)AccumulatedArea >= Config.Area) {
                        // std::cout << "\t Hit area target" << std::endl;
                        EarlyStop = true;
                        break;
                    }
                }
            }

            if (EarlyStop) break;

            // propagate wavefront
            int Deg = m_G.NumAdjacents(v);
            for (int j = 0; j < Deg; ++j) {
                auto Neig = m_G.GetAdjacent(v, j);

                double Length = Neig.second;
                double NewDist = Distances[v] + Length;

                if (NewDist < Distances[Neig.first]) {
                    Distances[Neig.first] = NewDist;
                    Queue.emplace(Distances[Neig.first], Neig.first);
                    // std::cout << "\t Queued vertices: " << Queue.size() << std::endl;
                }
            }
        }
        
        Steps++;
        std::cout << "\r\tCompletion: " << ((float)Steps / (float)Mesh.NumVertices()) * 100.f << "%..." << std::flush;
    }
    
    int Sum = 0, Min = std::numeric_limits<int>::max(), Max = 0;
    for (int i = 0; i < Mesh.NumVertices(); i++) {
        auto Size = PerVertexPartitions[i].size();
        Sum += Size;
        if (Size < Min) Min = Size;
        if (Size > Max) Max = Size;
    }

    std::cout << "\nPatch size stats:" << std::endl;
    std::cout << "\t\tMean: " << ((float)Sum / Mesh.NumVertices()) << " Min: " << Min << " Max: " << Max << std::endl;

    std::cout << std::endl << "Exporting... " << std::endl;
    if (!rmt::ExportVoronoi(Config.OutFile, PerVertexPartitions, FacesPerPartition, Faces)) {
        std::cerr << "Cannot write output." << std::endl;
        return -1;
    }
    if (Config.Update) {
        if (!rmt::ExportMesh(Config.InFile, Mesh.GetVertices(), Mesh.GetTriangles())) {
            std::cerr << "Cannot write output." << std::endl;
            return -1;
        }
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
    Args.OutFile = "";
    Args.Area = 1.f/1000.f;
    Args.Update = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string argvi(argv[i]);
        if (argvi == "-a" || argvi == "--area")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.Area = std::stof(argv[++i]);
            continue;
        }
        if (argvi == "-o" || argvi == "--output")
        {
            if (i == argc - 1)
            {
                Usage(argv[0], true);
            }
            Args.OutFile = argv[++i];
            continue;
        }
        if (argvi == "-u" || argvi == "--update")
        {
            Args.Update = true;
            continue;
        }
        if (Args.InFile.empty()) Args.InFile = argvi;
    }

    if (Args.InFile.empty())
    {
        std::cerr << "No input file given." << std::endl;
        Usage(argv[0], true);
    }
    if (Args.OutFile.empty())
    {
        auto InPath = std::filesystem::path(Args.InFile);
        Args.OutFile = (InPath.parent_path() / InPath.stem()).string() + "_regions.bin";
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
    out << "\t" << Prog << " in_file [-o|--output out_dir] [-a|--area area] [-u|--update]" << std::endl;
    out << "\t" << Prog << " -h|--help" << std::endl;
    out << std::endl;
    out << "Arguments details:" << std::endl;
    out << "\t- in_file is the input mesh file;" << std::endl;
    out << "\t- -a|--area sets the desired area for computed regions;" << std::endl;
    out << "\t- -o|--output sets the output file for the processed mesh;" << std::endl;
    out << "\t- -u|--update overwrites the input mesh with its repaired version;" << std::endl;
    out << "\t- -f|--file sets the arguments using the content of config_file." << std::endl;
    out << "\t- -h|--help prints this message." << std::endl;

    if (IsError)
        exit(-1);
}

