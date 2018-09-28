// author: joryschossau
// NOTE: requires data file in following format (notice the 3 new lines at top I added):
// (if numbers are 4 by 4, for example (usually they are 28 by 28))
// The first row is the dimension of the number data for a single digit
// Second row is how many classes and how many replicates per class
// Then the following repeating row format:
// Blank
// Class-Replicate
// DIGIT DATA MATRIX
// Blank
// Class-Replicate
// DIGIT DATA MATRIX
// etc.
/* $> cat example_file.mnist
4 4
10 100

5-1
0 1 1 1
0 1 1 0
0 0 0 1
0 1 1 0
*/
#include <iostream>
#include <iterator> // for istream_iterator, stream_iterator
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<unsigned short int> getSpaceSepNums(const std::string& line) {
    std::stringstream sline(line);
    return std::vector<unsigned short int>((std::istream_iterator<unsigned short int>(sline)), std::istream_iterator<unsigned short int>());
}

struct MNISTData {
    struct Point { int x,y; };
    Point dim; // dimension of a single digit (usually 28 x 28)
    int classes, reps; // number of classes (usually 10), number of replicates (usually 100)
    std::vector<std::vector<std::vector<std::vector<unsigned short int>>>> data;
    void load(const std::string& filename) {
        std::ifstream file;
        file.open(filename, std::ios::in);

        if (!file) {
            std::cout << "file '" << filename << "' doesn't exist" << std::endl;
            exit(1);
        }
        std::string line;
        std::getline(file, line);
        std::vector<unsigned short int> dimXY = getSpaceSepNums(line);
        dim.x = dimXY[0]; dim.y = dimXY[1];
        std::getline(file, line);
        std::vector<unsigned short int> classesAndReps = getSpaceSepNums(line);
        classes = classesAndReps[0]; reps = classesAndReps[1];
        /// data format [class][rep][row][col]
        data.resize(classes, std::vector<std::vector<std::vector<unsigned short int>>>(reps, std::vector<std::vector<unsigned short int>>(dim.y, std::vector<unsigned short int>(dim.x,0))));
        for (int numberi=classes*reps-1; numberi>=0; numberi--) { // for each number
            std::getline(file, line); // empty line
            std::getline(file, line); // CLASS-REPLICATE
            int numClass = std::stoi( line.substr(0,line.rfind("-")) );
            int numRep = std::stoi( line.substr(line.rfind("-")+1) );
            for (int rowi=0; rowi<dim.y; rowi++) {
                std::getline(file, line);
                std::vector<unsigned short int> row = getSpaceSepNums(line); // row of data
                std::copy(std::begin(row), std::end(row), std::begin(data[numClass][numRep-1][rowi]));
            }
        }
        file.close();
    }
    const std::vector<std::vector<unsigned short int>> numberAndRep(int numClass, int numRep) {
        return data[numClass][numRep]; /// return replicate of specific number(class) Note: rep is 0-index based
    }
    struct Flat {
        Point dim;
        int classes, reps;
        unsigned short int *data;
    };
    const Flat getFlatVersion() { // TODO: params here for making it 2D...?
        Flat pack;
        pack.data = (unsigned short int*)malloc(classes*reps*dim.x*dim.y*sizeof(unsigned short int));
        pack.dim.x = dim.x; pack.dim.y = dim.y;
        pack.classes = classes; pack.reps = reps;
        for (int classi=0; classi<classes; classi++) {
            for (int repi=0; repi<reps; repi++) {
                for (int rowi=0; rowi<dim.y; rowi++) {
                    std::copy(std::begin(data[classi][repi][rowi]),
                              std::end(data[classi][repi][rowi]),
                              pack.data + (dim.x*dim.y*repi + \
                                         dim.x*dim.y*reps*classi + \
                                         dim.x*rowi) );
                }
            }
        }
        return pack; // Will return by move semantics (yay compiler!)
    }
};
