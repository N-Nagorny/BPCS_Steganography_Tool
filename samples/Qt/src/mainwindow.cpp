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

MainWindow::~MainWindow()
{
    delete ui;
}
