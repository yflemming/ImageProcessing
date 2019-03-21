'''
Created on Apr 10, 2014

@author: Yuon
'''
import numpy as np
import sys
import vtk 

class MyMesh:
    
    def ReadNodes(self, fileName):
        try:
            inFile = open(fileName, 'r')
        except IOError:
            print "IOError: File","\'"+fileName+"\'","was not located in given directory or path"
            sys.exit()
        
        dim = inFile.readline().split()
    
        nodeMatrix = []
        
        for line in inFile:
            coords = line.split()
            
            try:
                if len(coords)!=int(dim[1]):
                    raise BaseException  
            except :
                print 'BaseException: Incorrect number of node coordinates in ', fileName, ": ", line
                sys.exit()
                
            tempArray = []
            for coord in coords:
                tempArray.append(float(coord))
            
            nodeMatrix.append(tempArray)
        
        self.node = nodeMatrix   

    def ReadElements(self, fileName):
        try:
            inFile = open(fileName, 'r')
        except IOError:
            print "IOError: File","\'"+fileName+"\'","was not located in given directory or path"
            sys.exit()
        
        dim = inFile.readline().split()
    
        elementMatrix = []
        for line in inFile:
            coords = line.split()
            
            try:
                if len(coords)!=int(dim[1]):
                    print len(coords), dim[1]
                    raise BaseException  
            except :
                print 'BaseException: Incorrect number of element coordinates in ', fileName, ": ", line
                sys.exit()
            
            tempArray = []
            for coord in coords:
                tempArray.append(int(coord))
            
            elementMatrix.append(tempArray)
                            
        self.elems = elementMatrix

    def GetNodes(self):
        return self.node
        
    def GetElements(self):
        return self.elems
    
    def RenderWireframe(self):
        myUnGrid = vtk.vtkUnstructuredGrid()
        myPoints = vtk.vtkPoints()
        myCells = vtk.vtkCellArray()
        renderTri = False;
        renderTetra = False;
        
        #def pts
        if len(self.node[0])==2:
            renderTri = True;
        else:
            renderTetra = True
            
        if renderTri == True:
            #set points
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], 0.0))
        
            #cells
            for coords in self.elems:
                myCells.InsertNextCell(3)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
            
            myUnGrid.SetPoints(myPoints)
            print "Here"
            myUnGrid.SetCells(vtk.VTK_TRIANGLE, myCells)
    
                
        if renderTetra == True:
            #points
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], coords[2]))
            
            #cells    
            for coords in self.elems:
                myCells.InsertNextCell(4)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
                myCells.InsertCellPoint(coords[3])
                
            myUnGrid.SetPoints(myPoints)
            print "here"
            myUnGrid.SetCells(vtk.VTK_TETRA, myCells)
        
        #get point data
        myScalarsP = vtk.vtkFloatArray()
        myScalarsP.SetNumberOfComponents(1) 
        for coord in self.node: 
            myScalarsP.InsertNextTuple1(coord[0]**2 + coord[1]**2)
        myUnGrid.GetPointData().SetScalars(myScalarsP)
         
        #mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(myUnGrid)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(0,1)

        #actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
#         actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.7,0.2,0.1) # color of the object
        actor.GetProperty().SetSpecular(.5)  # amount of specular lighting
        actor.GetProperty().SetSpecularPower(50) # controls how fast the specular lighting falls to zero away from the specular direction - the higher this number the faster light will fall
        actor.GetProperty().SetDiffuse(.8) # amount of diffuse lighting

        #renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor)

        #rendering window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300,300)

        renWin.Render()
        
        # interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renWin)
        interactor.Initialize()
        interactor.Start()
        
        
    def DisplayNodalScalars(self,vec):
        try:
            if len(vec)==len(self.nodes):
                print len(vec), len(self.nodes)
                raise BaseException  
        except :
            print 'Input vector is not correct size.'
            sys.exit()
            

a = MyMesh()
a.ReadElements("elems2D.txt")
a.ReadNodes("nodes2D.txt")
a.RenderWireframe()

nodes = a.GetNodes()
test_scalars = []
for coord in nodes:
    test_scalars.append(coord[0]**2 + coord[1]**2)

a.DisplayNodalScalars(test_scalars)

