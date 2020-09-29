#pragma once

#include <opencv2/core/core.hpp>

#include <string>

namespace BPCS {
    struct EmbedStats {
        unsigned int x_max = 0;
        unsigned int y_max = 0;
        std::string stats;
    };

    EmbedStats make_embed_stats(unsigned int a, unsigned int b, std::string c);

    bool***** segment(bool**** bitPlanes, unsigned int image_side, unsigned int channels, unsigned int seg_side);
    unsigned int grayencode(unsigned int g);
    unsigned int graydecode(unsigned int gray);
    unsigned int max_segment_complexity(unsigned int cols, unsigned int rows);

    EmbedStats embed(cv::Mat* image_orig, unsigned int seg_side, double alpha, std::istream& is);
    EmbedStats extract(cv::Mat* image_orig, unsigned int seg_side, double alpha, std::ostream& os);
}
