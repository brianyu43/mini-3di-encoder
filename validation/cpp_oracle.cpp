// Test harness only. Links unmodified pinned Foldseek/Kerasify sources.
// Protected geometry methods are exposed through standard public inheritance.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "structureto3di.h"

struct GeometryOracle : StructureTo3DiBase {
    using StructureTo3DiBase::approxCBetaPosition;
    using StructureTo3DiBase::createResidueMask;
    using StructureTo3DiBase::findResiduePartners;
    using StructureTo3DiBase::replaceCBWithVirtualCenter;
};

double read_number(std::istream& in) {
    std::string token;
    if (!(in >> token)) throw std::runtime_error("Missing numeric input");
    return std::stod(token);
}
Vec3 read_vec(std::istream& in) {
    double x=read_number(in), y=read_number(in), z=read_number(in);
    return Vec3(x,y,z);
}
void vector_json(Vec3 v) {
    std::cout << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}
void tensor_json(Tensor& tensor) {
    std::cout << '[';
    for (size_t k=0; k<tensor.data_.size(); ++k) {
        if (k) std::cout << ',';
        std::cout << tensor.data_[k];
    }
    std::cout << ']';
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("mode weights input [indices]");
        std::ifstream weights(argv[2], std::ios::binary), input(argv[3]);
        if (!weights || !input) throw std::runtime_error("Input open failed");
        std::string blob((std::istreambuf_iterator<char>(weights)), {});
        KerasModel model;
        if (!model.LoadModel(blob)) throw std::runtime_error("Model load failed");
        std::cout << std::setprecision(17);
        if (std::string(argv[1]) == "features") {
            size_t count;
            input >> count;
            if (count > 1024) throw std::runtime_error("Probe budget");
            std::cout << '[';
            for (size_t q=0; q<count; ++q) {
                Tensor features(10), embedding;
                for (int k=0; k<10; ++k) features.data_[k] = static_cast<float>(read_number(input));
                if (!model.Apply(&features, &embedding)) throw std::runtime_error("Apply failed");
                if (q) std::cout << ',';
                tensor_json(embedding);
            }
            std::cout << "]\n";
            return 0;
        }
        if (std::string(argv[1]) != "coordinates") throw std::runtime_error("Unknown mode");
        size_t count;
        input >> count;
        if (count < 3 || count > 10000) throw std::runtime_error("Chain size outside harness scope");
        std::vector<Vec3> n(count), ca(count), c(count), cb(count);
        for (size_t k=0; k<count; ++k) {
            n[k]=read_vec(input); ca[k]=read_vec(input); c[k]=read_vec(input); cb[k]=read_vec(input);
        }
        GeometryOracle geometry;
        auto virtuals = cb;
        std::vector<bool> mask(count);
        std::vector<int> partners(count, -1);
        geometry.createResidueMask(mask, ca.data(), n.data(), c.data(), count);
        geometry.replaceCBWithVirtualCenter(ca.data(), n.data(), c.data(), virtuals.data(), count);
        geometry.findResiduePartners(partners, virtuals.data(), mask, count);
        auto cb_for_full_call = cb;
        StructureTo3Di encoder;
        char* states = encoder.structure2states(ca.data(), n.data(), c.data(), cb_for_full_call.data(), count);
        auto features = encoder.getFeatures();
        std::cout << '[';
        for (int arg=4; arg<argc; ++arg) {
            int i=std::stoi(argv[arg]);
            if (i<0 || static_cast<size_t>(i)>=count) throw std::runtime_error("Invalid index");
            if (arg>4) std::cout << ',';
            int j = partners[i];
            std::cout << "{\"index\":" << i << ",\"partner\":" << j
                      << ",\"state\":" << static_cast<int>(states[i]);
            // The experiment uses a clean, complete backbone. Endpoint states are invalid.
            if (i==0 || static_cast<size_t>(i)==count-1) {
                std::cout << ",\"endpoint\":true}";
                continue;
            }
            if (j<0) throw std::runtime_error("Missing partner outside clean oracle scope");
            std::cout << ",\"virtual_center\":"; vector_json(virtuals[i]);
            std::cout << ",\"partner_virtual_center\":"; vector_json(virtuals[j]);
            Vec3 used=cb[i];
            if (std::isnan(used.x)) used=geometry.approxCBetaPosition(ca[i],n[i],c[i]);
            std::cout << ",\"cb_used\":"; vector_json(used);
            std::cout << ",\"features\":[";
            Tensor x(10), z;
            for (int k=0; k<10; ++k) {
                if (k) std::cout << ',';
                std::cout << features[i].f[k];
                x.data_[k]=static_cast<float>(features[i].f[k]);
            }
            if (!model.Apply(&x,&z)) throw std::runtime_error("Apply failed");
            std::cout << "],\"embedding\":"; tensor_json(z);
            std::cout << '}';
        }
        std::cout << "]\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
