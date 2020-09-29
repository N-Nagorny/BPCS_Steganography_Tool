#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
namespace Ui {
class MainWindow;
}

enum class ImgFormat {
    PNG,
    PGM
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
protected:

private slots:
    void openImage();
    void updateSpinBoxValue(double newValue);

private:
    Ui::MainWindow *ui;
    Mat   imagerd;
    Mat image_orig;
    unsigned int seg_side = 8;
    double alpha = 0.3;
    ImgFormat imgFormat = ImgFormat::PNG;

    Mat KochEmbedder(unsigned int seg_side, double alpha, QString message_name);
    Mat KochExtractor(unsigned int seg_side, double alpha);
    bool** calculateHashes(Mat image, unsigned int seg_side);
    void showImage(Mat image);
    void saveImage(Mat image, QString outputPath, ImgFormat format);
};

#endif // MAINWINDOW_H
