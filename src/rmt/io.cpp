/**
 * @file        io.cpp
 * 
 * @brief       Implements I/O functionalities.
 * 
 * @author      Filippo Maggioli\n
 *              (maggioli@di.uniroma1.it, maggioli.filippo@gmail.com)\n
 *              Sapienza, University of Rome - Department of Computer Science
 * 
 * @date        2023-10-26
 */
#include <rmt/io.hpp>

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <fstream>

#include <unsupported/Eigen/SparseExtra>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>

#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/writePLY.h>

bool rmt::ExportWeightmap(const std::string & Filename, 
                          const Eigen::SparseMatrix<double>& WM)
{
    return Eigen::saveMarket(WM, Filename);
}


bool rmt::LoadMesh(const std::string& Filename,
                   Eigen::MatrixXd& V,
                   Eigen::MatrixXi& F)
{
    std::string Ext;
    Ext = std::filesystem::path(Filename).extension().string();
    std::transform(Ext.begin(), Ext.end(), Ext.begin(), [](int c) { return std::tolower(c); });
    
    if (Ext == ".obj")
        return igl::readOBJ(Filename, V, F);
    else if (Ext == ".off")
        return igl::readOFF(Filename, V, F);
    else if (Ext == ".ply")
        return igl::readPLY(Filename, V, F);

    return false;
}


bool rmt::ExportMesh(const std::string & Filename, 
                     const Eigen::MatrixXd & V, 
                     const Eigen::MatrixXi & F)
{
    std::string Ext;
    Ext = std::filesystem::path(Filename).extension().string();
    std::transform(Ext.begin(), Ext.end(), Ext.begin(), [](int c) { return std::tolower(c); });

    if (Ext == ".obj")
        return igl::writeOBJ(Filename, V, F);
    else if (Ext == ".off")
        return igl::writeOFF(Filename, V, F);
    else if (Ext == ".ply")
        return igl::writePLY(Filename, V, F);
    
    return false;
}

bool rmt::ExportMesh(const std::string & Filename, 
                     const Eigen::MatrixXd & V, 
                     const Eigen::MatrixXi & F,
                     const Eigen::MatrixXd & Feats)
{
    std::string Ext;

    Eigen::MatrixXd _DummyDouble(0,0);
    Eigen::MatrixXi _DummyInt(0,0);
    std::vector<std::string> _dummy_header;

    return igl::writePLY(Filename, V, F, 
                         _DummyDouble, _DummyDouble, _DummyInt, 
                         Feats, {"VoronoiRegion", "IsCenter"}, _dummy_header);

}

bool rmt::ExportVoronoi(const std::string& Filename,
                        const std::vector<std::vector<unsigned int>>& VertexRegions,
                        const std::vector<std::vector<unsigned int>>& FaceRegions,
                        const Eigen::MatrixXi& Faces) 
{
    std::ofstream OutFile(Filename, std::ios::out | std::ios::binary);
    if (!OutFile.is_open()) 
        return false;

    // 1. Write the size of the outer vector
    size_t OuterSize = VertexRegions.size();
    OutFile.write(reinterpret_cast<const char*>(&OuterSize), sizeof(OuterSize));

    // 2. Iterate through each inner vector
    for (int i = 0; i < OuterSize; i++) {
        auto VRegion = VertexRegions[i];

        // a. Write the number of vertices
        size_t NumVertices = VRegion.size();
        OutFile.write(reinterpret_cast<const char*>(&NumVertices), sizeof(NumVertices));

        // b. Write the raw data of the inner vector
        if (NumVertices > 0) 
            OutFile.write(reinterpret_cast<const char*>(VRegion.data()), NumVertices * sizeof(int));

        // c. Write the number of faces
        auto FRegion = FaceRegions[i];
        size_t NumFaces = FRegion.size();
        OutFile.write(reinterpret_cast<const char*>(&NumFaces), sizeof(NumFaces));

        // d. Remap and write faces
        std::unordered_map<unsigned int, unsigned int> Map(NumVertices);
        for (int j = 0; j < NumVertices; j++) 
            Map.emplace(VRegion[j], j);
        
        std::vector<unsigned int> FaceRemap(NumFaces * 3);
        for (int j = 0; j < NumFaces; j++) {
            auto Face = Faces.row(j);
            FaceRemap.push_back(Map[Face[0]]);
            FaceRemap.push_back(Map[Face[1]]);
            FaceRemap.push_back(Map[Face[2]]);
        } 

        if (NumFaces > 0) 
            OutFile.write(reinterpret_cast<const char*>(FaceRemap.data()), NumFaces * 3 * sizeof(int));

    }

    OutFile.close();
    return true;
}