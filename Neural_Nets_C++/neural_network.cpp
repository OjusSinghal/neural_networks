#include <iostream>
#include <fstream>
#include <vector>

int main()
{
    // open image training set
    std::ifstream infile;
    infile.open("train-images.idx3-ubyte", std::ios::in | std::ios::binary);
    if (!infile) {
        std::cerr << "Could not open the file." << std::endl;
        return 5;
    }

    // read off header bytes
    const int HEADER_BYTES = 16;
    for (int i = 0; i < HEADER_BYTES; i++) {
        char magic_number;
        infile.read(&magic_number, sizeof(char));
        // std::cout << i << ' ' << (int)magic_number << std::endl;
    }
    const int ROW_COUNT = 28;
    const int COLUMN_COUNT = 28;
    const int IMAGE_COUNT = 60000;

    std::vector<std::vector<char>> images(IMAGE_COUNT, std::vector<char>(ROW_COUNT * COLUMN_COUNT));
    for (int image = 0; image < IMAGE_COUNT; image++) {
        for (int pixel = 0; pixel < ROW_COUNT * COLUMN_COUNT; pixel++) {
            infile.read(&images[image][pixel], sizeof(char));
        }
    }


}