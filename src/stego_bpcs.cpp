#include "stego_bpcs.h"

bool***** BPCS::segment(bool**** bitPlanes, unsigned int image_side, unsigned int channels, unsigned int seg_side) {
    int Nc = image_side * image_side / (seg_side*seg_side);

    bool***** segments = new bool****[channels];
    for (int i = 0; i < channels; i++) {
        segments[i] = new bool***[8];
        for (int j = 0; j < 8; j++) {
            segments[i][j] = new bool**[Nc];
            for (int k = 0; k < Nc; k++) {
                segments[i][j][k] = new bool*[seg_side];
                for (int l = 0; l < seg_side; l++) {
                    segments[i][j][k][l] = new bool[seg_side];
                }
            }
        }
    }

    for (int l = 0; l < channels; l++) {
        for (int m = 0; m < 8; m++) {

            int y = 0;
            int n = 0;
            int x;
            while (y < image_side) {
                x = 0;
                while (x < image_side) {
                    for (int i = 0; i < seg_side; i++) {
                        for (int j = 0; j < seg_side; j++) {
                            segments[l][m][n][i][j] = bitPlanes[x + i][ y + j][l][m];
                        }
                    }
                    x = x + seg_side;
                    n = n + 1;
//                    qDebug() << n << '/' << Nc;
                }
                y = y + seg_side;
            }
        }
    }
    return segments;
}

unsigned int BPCS::grayencode(unsigned int g)
{
    return g ^ (g >> 1);
}
unsigned int BPCS::graydecode(unsigned int gray)
{
    unsigned int bin;
    for (bin = 0; gray; gray >>= 1) {
      bin ^= gray;
    }
    return bin;
}

unsigned int BPCS::max_segment_complexity(unsigned int cols, unsigned int rows) {
    return ((rows-1)*cols) + ((cols-1)*rows);
}

BPCS::EmbedStats BPCS::make_embed_stats(unsigned int a, unsigned int b, std::string c) {
    BPCS::EmbedStats f;
    f.x_max = a;
    f.y_max = b;
    f.stats = c;
    return f;
}