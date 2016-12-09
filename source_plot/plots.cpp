#include "plots.hpp"

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkMarchingCubes.h>
#include <vtkVoxelModeller.h>
#include <vtkSphereSource.h>
#include <vtkImageData.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>

#include <vtkActor.h>
#include <vtkDelaunay2D.h>
#include <vtkLookupTable.h>
#include <vtkMath.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkXMLPolyDataWriter.h>

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif


void plot3D(std::vector<int> size, std::vector<double> target, double isoValue)
{
    vtkSmartPointer<vtkImageData> volume =
            vtkSmartPointer<vtkImageData>::New();
    int dims[3];
    dims[0] = size[0];
    dims[1] = size[1];
    dims[2] = size[2];
    volume->SetDimensions(dims);
#if VTK_MAJOR_VERSION <= 5
    volume->SetNumberOfScalarComponents(1);
  volume->SetScalarTypeToDouble();
#else
    volume->AllocateScalars(VTK_DOUBLE,1);
#endif

    for (int i = 0; i < size[0]; i++)
    {
        for (int j = 0; j < size[1]; j++)
        {
            for (int k = 0; k < size[2]; k++)
            {
                double* pixel = static_cast<double*>(volume->GetScalarPointer(i,j,k));
                pixel[0] = target[i + j*size[0] + k*size[0]*size[1]];
            }
        }
    }

    vtkSmartPointer<vtkMarchingCubes> surface =
            vtkSmartPointer<vtkMarchingCubes>::New();

#if VTK_MAJOR_VERSION <= 5
    surface->SetInput(volume);
#else
    surface->SetInputData(volume);
#endif
    surface->ComputeNormalsOn();
    surface->SetValue(0, isoValue);

    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(.1, .2, .3);

    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> interactor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(surface->GetOutputPort());
    mapper->ScalarVisibilityOff();

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    renderer->AddActor(actor);

    renderWindow->Render();
    interactor->Start();
}

void plot2D(std::vector<int> size, std::vector<double> target){

// Create a grid of points (height/terrian map)
    vtkSmartPointer<vtkPoints> points =
            vtkSmartPointer<vtkPoints>::New();

    for(int i = 0; i < size[0]; i++)
    {
        for(int j = 0; j < size[1]; j++)
        {
            points->InsertNextPoint(i, j, target[i + j*size[0]]);
        }
    }

    // Add the grid points to a polydata object
    vtkSmartPointer<vtkPolyData> inputPolyData =
            vtkSmartPointer<vtkPolyData>::New();
    inputPolyData->SetPoints(points);

    // Triangulate the grid points
    vtkSmartPointer<vtkDelaunay2D> delaunay =
            vtkSmartPointer<vtkDelaunay2D>::New();
#if VTK_MAJOR_VERSION <= 5
    delaunay->SetInput(inputPolyData);
#else
    delaunay->SetInputData(inputPolyData);
#endif
    delaunay->Update();
    vtkPolyData* outputPolyData = delaunay->GetOutput();

    double bounds[6];
    outputPolyData->GetBounds(bounds);

    // Find min and max z
    double minz = bounds[4];
    double maxz = bounds[5];

    // Create the color map
    vtkSmartPointer<vtkLookupTable> colorLookupTable =
            vtkSmartPointer<vtkLookupTable>::New();
    colorLookupTable->SetTableRange(minz, maxz);
    colorLookupTable->Build();

    // Generate the colors for each point based on the color map
    vtkSmartPointer<vtkUnsignedCharArray> colors =
            vtkSmartPointer<vtkUnsignedCharArray>::New();
    colors->SetNumberOfComponents(3);
    colors->SetName("Colors");

    for(int i = 0; i < outputPolyData->GetNumberOfPoints(); i++)
    {
        double p[3];
        outputPolyData->GetPoint(i,p);

        double dcolor[3];
        colorLookupTable->GetColor(p[2], dcolor);
        unsigned char color[3];
        for(unsigned int j = 0; j < 3; j++)
        {
            color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
        }

        colors->InsertNextTupleValue(color);
    }

    outputPolyData->GetPointData()->SetScalars(colors);

    // Create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
#if VTK_MAJOR_VERSION <= 5
    mapper->SetInputConnection(outputPolyData->GetProducerPort());
#else
    mapper->SetInputData(outputPolyData);
#endif

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(.1, .2, .3);

    // Render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
}