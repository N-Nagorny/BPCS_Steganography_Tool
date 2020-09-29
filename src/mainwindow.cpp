#include "mainwindow.h"
#include "stego_bpcs.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <fstream>
#include <set>
#include <QDebug>
#include <QStringList>

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->spinBox->setMinimum(0);
    ui->spinBox->setMaximum(1);
    ui->spinBox->setValue(0.3);
    ui->spinBox->setSingleStep(0.1);
    QObject::connect(ui->pushButton, SIGNAL(clicked()), this, SLOT(openImage()));
    QObject::connect(ui->spinBox, SIGNAL(valueChanged(double)), this, SLOT(updateSpinBoxValue(double))); //"Обзор..." button
}

void MainWindow::updateSpinBoxValue(double newValue) {
    alpha = newValue;
}

//Open image function call
void MainWindow::openImage()
{
    QString container_name = QFileDialog::getOpenFileName(this,
                                    tr("Open a container"), QDir::currentPath(),tr("Image Files (*.png *.pgm *.jpg *.jpeg *.bmp)"));

    if( container_name.size() )
    {
        image_orig = imread(container_name.toStdString().c_str());

        showImage(image_orig);
        QFileInfo info(container_name);
        if (ui->radioButton->isChecked()) {
            QString message_name = QFileDialog::getOpenFileName(this, tr("Open a message"), QDir::currentPath());
            QString strNewName = info.path() + "/" + info.completeBaseName() + "_emb";

            ifstream is;
            is.open(message_name.toStdString());
            BPCS::EmbedStats stats = BPCS::embed(&image_orig, seg_side, alpha, is);
            is.close();

            ui->statusBar->showMessage(QString::fromStdString(stats.stats));

            Mat rectangled = image_orig.clone();
            rectangle(rectangled , Rect(0, 0, stats.x_max, stats.y_max), Scalar(255), 1, 8, 0);
            showImage(rectangled);

            saveImage(image_orig, strNewName, imgFormat);
        }
        else if (ui->radioButton_2->isChecked()) {
            QString strNewName = info.path() + "/" + info.completeBaseName() + "_ext";

            ofstream os;
            os.open("out.txt");
            BPCS::EmbedStats stats = BPCS::extract(&image_orig, seg_side, alpha, os);
            os.close();

            Mat rectangled = image_orig.clone();
            rectangle(rectangled , Rect(0, 0, stats.x_max, stats.y_max), Scalar(255), 1, 8, 0);
            showImage(rectangled);

            saveImage(image_orig, strNewName, imgFormat);
        }
    }
}

void MainWindow::showImage(Mat image) {
    Mat rgb;
    cvtColor(image, rgb, CV_BGR2RGB);

    QImage qimage = QImage((const unsigned char*)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    QGraphicsScene* scene = new QGraphicsScene();
    scene->addPixmap(QPixmap::fromImage(qimage));
    ui->graphicsView->setScene(scene);
}

BPCS::EmbedStats BPCS::embed(Mat* image_orig, unsigned int seg_side, double alpha, istream& is) {
    bool** Wc = new bool*[seg_side];
    for (unsigned int i = 0; i < seg_side; i++) {
        Wc[i] = new bool[seg_side];
    }
    for (int i = 0; i < seg_side; i++)
        for (int j = 0; j < seg_side; j++)
            Wc[i][j] = ((i % 2 != 0 && j % 2 == 0) || (i % 2 == 0 && j % 2 != 0)) ? true : false;

    unsigned int x_max = seg_side;
    unsigned int y_max = seg_side;
    while (x_max + seg_side <= image_orig->cols && y_max + seg_side <= image_orig->rows) {
        x_max += seg_side;
        y_max += seg_side;
    }
    Rect Rec(0, 0, x_max, y_max);

    Mat image(*image_orig, Rec);
// PBC to CGC conversion
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for(int k = 0; k < image.channels(); k++) {
                image.at<Vec3b>(i, j)[k] = BPCS::grayencode(image.at<Vec3b>(i, j)[k]);
            }
        }
    }
// Image channels bitwise splitting
    bool**** bitPlanes = new bool***[image.rows];
    for (int i = 0; i < image.rows; i++) {
        bitPlanes[i] = new bool**[image.cols];
        for (int j = 0; j < image.cols; j++) {
            bitPlanes[i][j] = new bool*[image.channels()];
            for (int k = 0; k < image.channels(); k++) {
                bitPlanes[i][j][k] = new bool[8];
            }
        }
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                for (int l = 0; l < 8; l++) {
                    bitPlanes[i][j][k][l] = (image.at<Vec3b>(i, j)[k] >> l) & 1U;
                }
            }
        }
    }
// Calculating container segments complexity
    int Nc = image.rows * image.cols / (seg_side*seg_side);
//qDebug() << image.channels();
    bool***** segments = BPCS::segment(bitPlanes,image.rows, image.channels(), seg_side);

    bool*** iscomplex = new bool**[Nc];
    for (int i = 0; i < Nc; i++) {
        iscomplex[i] = new bool*[image.channels()];
        for (int j = 0; j < image.channels(); j++) {
            iscomplex[i][j] = new bool[8];
        }
    }

    unsigned int maxComplexBlocks = 0;
    for (int k = 0; k < image.channels(); k++) {
        for (int m = 0; m < Nc; m++) {

            for (int l = 0; l < 8; l++) {

                float max_complexity = (float) BPCS::max_segment_complexity(seg_side, seg_side);
                int counter = 0;
                float complexity;
                for (int i = 0; i < seg_side; i++) {
                    for (int j = 0; j < seg_side; j++) {
                        if (i != 7 && segments[k][l][m][i][j] != segments[k][l][m][i+1][j])
                            counter++;
                        if (j != 7 && segments[k][l][m][i][j] != segments[k][l][m][i][j+1])
                            counter++;
                    }
                }
//            for (int j = 0; j < 8; j++) {
//                for (int i = 1; i < 8; i++) {
//                    complexity += segments[k][l][m][i][j] ^ segments[k][l][m][i-1][j];
//                }
//            }

                complexity = counter / max_complexity;

                complexity >= alpha ? (iscomplex[m][k][l] = true, maxComplexBlocks++) : iscomplex[m][k][l] = false;
                qDebug() << complexity << ';' << iscomplex[m][k][l];
            }
        }
    }

// Secret message segmenting
    is.seekg (0, is.end);
    long length = is.tellg();
    is.seekg (0, is.beg);
    long message_blocks = (long)length / 8 == (float)length / 8 ? length / 8 + 1 : (long)length / 8 + 2;
    long conjugation_map_blocks = (long)message_blocks / 64 == (float)message_blocks / 64 ? message_blocks / 64 : (long)message_blocks / 64 + 1;
//            message_blocks += conjugation_map_blocks;
    qDebug() << (long)conjugation_map_blocks/ 64;
    qDebug() << (float)conjugation_map_blocks/ 64;
    qDebug() << (long)length / 8;
    qDebug() << (float)length / 8;

    bool*** message = new bool**[message_blocks];
    for (int i = 0; i < message_blocks; i++) {
        message[i] = new bool*[seg_side];
        for (int j = 0; j < seg_side; j++) {
            message[i][j] = new bool[seg_side];
        }
    }

    bool*** conjugation_map = new bool**[conjugation_map_blocks ];
    for (int i = 0; i < conjugation_map_blocks ; i++) {
        conjugation_map [i] = new bool*[seg_side];
        for (int j = 0; j < seg_side; j++) {
            conjugation_map [i][j] = new bool[seg_side];
        }
    }

    for (int i = 0; i < seg_side; i++) {
        for (int j = 0; j < seg_side; j++) {
        message[0][i][j] = (length >> i*seg_side+j) & 1U;
        }
    }

    for (int batch_num = 1; batch_num < message_blocks; batch_num++ ) {
        for (int i = 0; i < seg_side; i++) {
            if (is.tellg() == length) {
                i = seg_side;
                break;
            }
            char buffer;
            is.read (&buffer,1);
            qDebug() << buffer;
            for (int j = 0; j < seg_side; j++) {
                message[batch_num][i][j] = (buffer >> j) & 1U;
            }
        }
    }
    qDebug() << "batch_num: " << message_blocks<< " conjugation_map_blocks: " << conjugation_map_blocks;

// Calculating segments complexity для сообщения
    bool* isConjugated = new bool[message_blocks - 1];

//    isConjugated[0] = false;
    for (int k = 1; k < message_blocks; k++) {
        float max_complexity = (float) BPCS::max_segment_complexity(seg_side, seg_side);
        int counter = 0;
        float complexity;
        for (int i = 0; i < seg_side; i++) {
            for (int j = 0; j < seg_side; j++) {
                if (i != 7 && message[k][i][j] != message[k][i+1][j])
                    counter++;
                if (j != 7 && message[k][i][j] != message[k][i][j+1])
                    counter++;
            }
        }

        complexity = counter / max_complexity;

        if (complexity < alpha) {
            isConjugated[k - 1] = true;
            for (int i = 0; i < seg_side; i++) {
                for (int j = 0; j < seg_side; j++) {
                    message[k][i][j] ^= Wc[i][j];
                }
            }
        }
        else
            isConjugated[k - 1] = false;
    }
    int k = 0;
    for (int batch_num = 0; batch_num < conjugation_map_blocks; batch_num++ ) {
        for (int i = 0; i < seg_side; i++) {
            for (int j = 0; j < seg_side; j++) {
                if (k == message_blocks - 1) {
                    j = seg_side;
                    i = seg_side;
                    break;
                }
                else{
                    conjugation_map[batch_num][i][j] = isConjugated[k];
                    k++;
                }
            }
        }
    }
    ofstream os1;
    os1.open("con_emb.txt");
    for (int i_mes = 0; i_mes < message_blocks - 1; i_mes++) {
        os1 << isConjugated[i_mes] << ", ";
    }
    os1.close();


// Message length embedding
    segments[0][0][0] = message[0];

// Conjugation map embedding
    for (int m = 1; m < conjugation_map_blocks + 1; m++) {
            segments[0][0][m] = conjugation_map[m - 1];
    }

// Embedding
    int i_mes = 1;
    for (int k = 0; k < image.channels(); k++) {
        for (int m = conjugation_map_blocks + 1; m < Nc; m++) {
            for (int l = 0; l < 8; l++) {
                if (i_mes == message_blocks) {
                    k = image.channels();
                    m = Nc;
                    l = 8;
                    break;
                }
                if (iscomplex[m][k][l]) {
                    segments[k][l][m] = message[i_mes];
                    i_mes++;
                }
            }
        }
    }

// Building bitplanes from segments
    bool**** out_bitPlanes = new bool***[image.rows];
    for (int i = 0; i < image.rows; i++) {
        out_bitPlanes[i] = new bool**[image.cols];
        for (int j = 0; j < image.cols; j++) {
            out_bitPlanes[i][j] = new bool*[image.channels()];
            for (int k = 0; k < image.channels(); k++) {
                out_bitPlanes[i][j][k] = new bool[8];
            }
        }
    }

    for (int l = 0; l < image.channels(); l++) {
        for (int m = 0; m < 8; m++) {

            int y = 0;
            int n = 0;
            int x;
            while (y < image.rows) {
                x = 0;
                while (x < image.cols) {
                    for (int i = 0; i < seg_side; i++) {
                        for (int j = 0; j < seg_side; j++) {
                            out_bitPlanes[x + i][ y + j][l][m] = segments[l][m][n][i][j];
                        }
                    }
                    x = x + seg_side;
                    n = n + 1;
//                                qDebug() << n << '/' << Nc;
                }
                y = y + seg_side;
            }
        }
    }
    qDebug() << "desegment";

// Building pixels from bitplanes
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                unsigned char pixel = 0;
                for (int l = 0; l < 8; l++) {

                    pixel |= out_bitPlanes[i][j][k][l] << l;
                }
                image.at<Vec3b>(i, j)[k] = pixel;
            }
        }
    }

// CGC to PBC conversion
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for(int k = 0; k < image.channels(); k++) {
                image.at<Vec3b>(i, j)[k] = BPCS::graydecode(image.at<Vec3b>(i, j)[k]);
            }
        }
    }

// Heap-allocated variables deleting
// TODO: use some smarter than simple pointers to deal with segment() result

    for(int i = 0; i < seg_side; i++) {
        delete [] Wc[i];
    }
    delete [] Wc;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                delete [] bitPlanes[i][j][k];
            }
            delete [] bitPlanes[i][j];
        }
        delete [] bitPlanes[i];
    }
    delete [] bitPlanes;

    for (int i = 0; i < Nc; i++) {
        for (int j = 0; j < image.channels(); j++) {
            delete [] iscomplex[i][j];
        }
        delete [] iscomplex[i];
    }
    delete [] iscomplex;

    for (int i = 0; i < message_blocks; i++) {
        for (int j = 0; j < seg_side; j++) {
            delete [] message[i][j];
        }
        delete [] message[i];
    }
    delete [] message;


    delete [] isConjugated;


    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                delete [] out_bitPlanes[i][j][k];
            }
            delete [] out_bitPlanes[i][j];
        }
        delete [] out_bitPlanes[i];
    }
    delete [] out_bitPlanes;
    return BPCS::make_embed_stats(x_max, y_max,
                to_string(maxComplexBlocks) + '/' + to_string(image.channels()*Nc*8) + " blocks | " +
                to_string((float)maxComplexBlocks / ((float)image.channels()*Nc*8 / 100)) + '%' + " | " +
                to_string(maxComplexBlocks * 8) + '/' + to_string(image_orig->cols * image_orig->rows * image.channels()) + " bytes | " +
                to_string((float)maxComplexBlocks * 8 / ((float)image_orig->cols * image_orig->rows * image.channels() / 100)) + '%'
    );
}

void MainWindow::saveImage(Mat image, QString outputPath, ImgFormat format) {
    QString finalPath;
    Mat finalMat;
    switch(format) {
    case ImgFormat::PNG:
        finalPath = outputPath + ".png";
        finalMat = image;
        break;
    case ImgFormat::PGM:
        finalPath = outputPath + ".pgm";
        Mat grayImage;
        cvtColor(image, grayImage, CV_BGR2GRAY);
        finalMat = grayImage;
        break;
    }
    imwrite(finalPath.toStdString().c_str(), finalMat);
}

BPCS::EmbedStats BPCS::extract(cv::Mat* image_orig, unsigned int seg_side, double alpha, std::ostream& os) {
    bool** Wc = new bool*[seg_side];
    for (unsigned int i = 0; i < seg_side; i++) {
        Wc[i] = new bool[seg_side];
    }
    for (int i = 0; i < seg_side; i++)
        for (int j = 0; j < seg_side; j++)
            Wc[i][j] = ((i % 2 != 0 && j % 2 == 0) || (i % 2 == 0 && j % 2 != 0)) ? true : false;

    unsigned int x_max = seg_side;
    unsigned int y_max = seg_side;
    while (x_max + seg_side <= image_orig->cols && y_max + seg_side <= image_orig->rows) {
        x_max += seg_side;
        y_max += seg_side;
    }
    Rect Rec(0, 0, x_max, y_max);

    Mat image(*image_orig, Rec);
// PBC to CGC conversion
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for(int k = 0; k < image.channels(); k++) {
                image.at<Vec3b>(i, j)[k] = BPCS::grayencode(image.at<Vec3b>(i, j)[k]);
            }
        }
    }
// Image channels bitwise splitting
    bool**** bitPlanes = new bool***[image.rows];
    for (int i = 0; i < image.rows; i++) {
        bitPlanes[i] = new bool**[image.cols];
        for (int j = 0; j < image.cols; j++) {
            bitPlanes[i][j] = new bool*[image.channels()];
            for (int k = 0; k < image.channels(); k++) {
                bitPlanes[i][j][k] = new bool[8];
            }
        }
    }

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                for (int l = 0; l < 8; l++) {
                    bitPlanes[i][j][k][l] = (image.at<Vec3b>(i, j)[k] >> l) & 1U;
                }
            }
        }
    }
// Calculating container segments complexity
    int Nc = image.rows * image.cols / (seg_side*seg_side);
//qDebug() << image.channels();
    bool***** segments = BPCS::segment(bitPlanes,image.rows, image.channels(), seg_side);

    bool*** iscomplex = new bool**[Nc];
    for (int i = 0; i < Nc; i++) {
        iscomplex[i] = new bool*[image.channels()];
        for (int j = 0; j < image.channels(); j++) {
            iscomplex[i][j] = new bool[8];
        }
    }

    unsigned int maxComplexBlocks = 0;
    for (int k = 0; k < image.channels(); k++) {
        for (int m = 0; m < Nc; m++) {

            for (int l = 0; l < 8; l++) {

                float max_complexity = (float) BPCS::max_segment_complexity(seg_side, seg_side);
                int counter = 0;
                float complexity;
                for (int i = 0; i < seg_side; i++) {
                    for (int j = 0; j < seg_side; j++) {
                        if (i != 7 && segments[k][l][m][i][j] != segments[k][l][m][i+1][j])
                            counter++;
                        if (j != 7 && segments[k][l][m][i][j] != segments[k][l][m][i][j+1])
                            counter++;
                    }
                }
//            for (int j = 0; j < 8; j++) {
//                for (int i = 1; i < 8; i++) {
//                    complexity += segments[k][l][m][i][j] ^ segments[k][l][m][i-1][j];
//                }
//            }
                complexity = counter / max_complexity;
                complexity >= alpha ? (iscomplex[m][k][l] = true, maxComplexBlocks++) : iscomplex[m][k][l] = false;
            }
        }
    }

// Message length extracting
    unsigned long length = 0;
    for (int i = 0; i < seg_side; i++) {
        for (int j = 0; j < seg_side; j++) {
        length |= segments[0][0][0][i][j] << (i*seg_side+j);
        }
    }
    qDebug() << "read length: " << length;
    long message_blocks = (long)length / seg_side == (float)length / seg_side ? length / seg_side : (long)length / seg_side + 1;
    long conjugation_map_blocks = (long)message_blocks / (seg_side*seg_side) == (float)message_blocks / (seg_side*seg_side) ? message_blocks / (seg_side*seg_side) : (long)message_blocks / (seg_side*seg_side) + 1;

    qDebug() << "batch_num: " << message_blocks<< " conjugation_map_blocks: " << conjugation_map_blocks;

    bool*** conjugation_map = new bool**[conjugation_map_blocks ];
    for (int i = 0; i < conjugation_map_blocks ; i++) {
        conjugation_map [i] = new bool*[seg_side];
        for (int j = 0; j < seg_side; j++) {
            conjugation_map [i][j] = new bool[seg_side];
        }
    }

    bool*** message = new bool**[message_blocks];
    for (int i = 0; i < message_blocks; i++) {
        message[i] = new bool*[seg_side];
        for (int j = 0; j < seg_side; j++) {
            message[i][j] = new bool[seg_side];
        }
    }

// Conjugation map extracting
    for (int m = 1; m < conjugation_map_blocks + 1; m++) {
        conjugation_map[m - 1] = segments[0][0][m];
    }

// Extracting
    int i_mes = 0;
    for (int k = 0; k < image.channels(); k++) {
        for (int m = conjugation_map_blocks + 1; m < Nc; m++) {
            for (int l = 0; l < 8; l++) {
                if (i_mes == message_blocks) {
                    k = image.channels();
                    m = Nc;
                    l = 8;
                    break;
                }
                if (iscomplex[m][k][l]) {
                    message[i_mes] = segments[k][l][m];
                    i_mes++;
                }
            }
        }
    }

// Deconjugating
    bool* isConjugated = new bool[message_blocks];

    int k = 0;
    for (int batch_num = 0; batch_num < conjugation_map_blocks; batch_num++ ) {
        for (int i = 0; i < seg_side; i++) {
            for (int j = 0; j < seg_side; j++) {
                if (k == message_blocks) {
                    j = seg_side;
                    i = seg_side;
                    break;
                }
                else{
                    isConjugated[k] = conjugation_map[batch_num][i][j];
                    k++;
                }
            }
        }
    }

    ofstream os1;
    os1.open("con_ext.txt");
    for (int i_mes = 0; i_mes < message_blocks; i_mes++) {
        os1 << isConjugated[i_mes] << ", ";
    }
    os1.close();

    for (int k = 0; k < message_blocks; k++) {
        if (isConjugated[k]) {
            for (int i = 0; i < seg_side; i++) {
                for (int j = 0; j < seg_side; j++) {
                    message[k][i][j] ^= Wc[i][j];
                }
            }
        }
    }

// Secret message building from segments
    for (int i_mes = 0; i_mes < message_blocks; i_mes++) {
        for (int i = 0; i < 8; i++) {
            if (i_mes * seg_side +i< length) {
                char buffer = (char) 0;
                for (int j = 0; j < 8; j++) {
                    buffer |= message[i_mes][i][j] << j;
                }
            os << buffer;
            }
        }
    }

// Building bitplanes from segments
    bool**** out_bitPlanes = new bool***[image.rows];
    for (int i = 0; i < image.rows; i++) {
        out_bitPlanes[i] = new bool**[image.cols];
        for (int j = 0; j < image.cols; j++) {
            out_bitPlanes[i][j] = new bool*[image.channels()];
            for (int k = 0; k < image.channels(); k++) {
                out_bitPlanes[i][j][k] = new bool[8];
            }
        }
    }

    for (int l = 0; l < image.channels(); l++) {
        for (int m = 0; m < 8; m++) {

            int y = 0;
            int n = 0;
            int x;
            while (y < image.rows) {
                x = 0;
                while (x < image.cols) {
                    for (int i = 0; i < seg_side; i++) {
                        for (int j = 0; j < seg_side; j++) {
                            out_bitPlanes[x + i][ y + j][l][m] = segments[l][m][n][i][j];
                        }
                    }
                    x = x + seg_side;
                    n = n + 1;
//                                qDebug() << n << '/' << Nc;
                }
                y = y + seg_side;
            }
        }
    }
    qDebug() << "desegment";

// Building pixels from bitplanes
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                unsigned char pixel = 0;
                for (int l = 0; l < 8; l++) {

                    pixel |= out_bitPlanes[i][j][k][l] << l;
                }
                image.at<Vec3b>(i, j)[k] = pixel;
            }
        }
    }

// CGC to PBC conversion
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for(int k = 0; k < image.channels(); k++) {
                image.at<Vec3b>(i, j)[k] = BPCS::graydecode(image.at<Vec3b>(i, j)[k]);
            }
        }
    }

// Heap-allocated variables deleting
// TODO: use some smarter than simple pointers to deal with segment() result

    for(int i = 0; i < seg_side; i++) {
        delete [] Wc[i];
    }
    delete [] Wc;


    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                delete [] bitPlanes[i][j][k];
            }
            delete [] bitPlanes[i][j];
        }
        delete [] bitPlanes[i];
    }
    delete [] bitPlanes;


    for (int i = 0; i < Nc; i++) {
        for (int j = 0; j < image.channels(); j++) {
            delete [] iscomplex[i][j];
        }
        delete [] iscomplex[i];
    }
    delete [] iscomplex;

    for (int i = 0; i < message_blocks; i++) {
        for (int j = 0; j < seg_side; j++) {
            delete [] message[i][j];
        }
        delete [] message[i];
    }
    delete [] message;


    delete [] isConjugated;


    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.channels(); k++) {
                delete [] out_bitPlanes[i][j][k];
            }
            delete [] out_bitPlanes[i][j];
        }
        delete [] out_bitPlanes[i];
    }
    delete [] out_bitPlanes;
    return BPCS::make_embed_stats(x_max, y_max, "");
}

MainWindow::~MainWindow()
{
    delete ui;
}
