// TestPyTorch.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <torch/script.h> 

#include <iostream>
#include <memory>
#include <VisionHeader.h>
#include <opencv2/highgui.hpp>
//#include <filesystem>

//namespace fs = std::filesystem;

std::string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    std::cout << "############### transpose ############" << std::endl;
    std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    std::cout << "shape after : " << tensor.sizes() << std::endl;
    std::cout << "######################################" << std::endl;
    return tensor;
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);

    tensor_image = tensor_image.permute({ 2, 0, 1 });

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }
    
    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }

    std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

auto ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

auto ToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        
        show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    const std::string strModelFile(argv[1]);

    //if (!fs::exists(strModelFile)) {
    //    std::cout << "File " << strModelFile << " not exists" << std::endl;
    //    return -1;
    //}
    std::cout << "File " << strModelFile << std::endl;

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(strModelFile.c_str());

        cv::Mat matRead = cv::imread("./TestImg/good_image_10.png", cv::ImreadModes::IMREAD_COLOR);
        if (matRead.empty()) {
            std::cout << "Failed to read image" << std::endl;
            return -1;
        }

        //auto testTensor = torch::ones({1, 28, 28});
        //std::cout << "test tenor shape: " << testTensor.sizes() << std::endl;

        auto tensor = ToTensor(matRead, true, true);
        std::cout << "test tenor shape: " << tensor.sizes() << std::endl;

        std::cout << "After permute, test tenor shape: " << tensor.sizes() << std::endl;


        tensor = tensor.toType(c10::kFloat).div(255);


        std::vector<torch::jit::IValue> inputs;
        //inputs.push_back(testTensor);
        inputs.push_back(tensor);

        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/100) << '\n';
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
