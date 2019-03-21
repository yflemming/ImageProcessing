'''
Created on Apr 10, 2014

@author: Yuon
'''

import numpy as np
import sys
import vtk 
from pysparse.sparse import spmatrix
from pysparse.direct import superlu

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
        Tri = False;
        Tetra = False;
        
        #define pts
        if len(self.node[0])==2:
            Tri = True;
        else:
            Tetra = True
            
        if Tri == True:
            #set pts
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], 0.0))
        
            #set Cells
            for coords in self.elems:
                myCells.InsertNextCell(3)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
            
            myUnGrid.SetPoints(myPoints)
            print "Here"
            myUnGrid.SetCells(vtk.VTK_TRIANGLE, myCells)
    
                
        if Tetra == True:
            #set pts
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], coords[2]))
            
            #set Cells    
            for coords in self.elems:
                myCells.InsertNextCell(4)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
                myCells.InsertCellPoint(coords[3])
                
            myUnGrid.SetPoints(myPoints)
            print "here"
            myUnGrid.SetCells(vtk.VTK_TETRA, myCells)
        
        #mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(myUnGrid)

        #actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()

        #renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor)

        #render window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300,300)

        renWin.Render()
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renWin)
        interactor.Initialize()
        interactor.Start()
        
        
    def DisplayNodalScalars(self, vector):
        try:
            if len(vector)!=len(self.node):
                raise BaseException
        except:
            print "Input vector length does not length of nodes in class"
            sys.exit()
            
        myUnGrid = vtk.vtkUnstructuredGrid()
        myPoints = vtk.vtkPoints()
        myCells = vtk.vtkCellArray()
        Tri = False;
        Tetra = False;
        
        #define pts
        if len(self.node[0])==2:
            Tri = True;
        else:
            Tetra = True
            
        if Tri == True:
            #set pts
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], 0.0))
        
            #set Cells
            for coords in self.elems:
                myCells.InsertNextCell(3)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
            
            myUnGrid.SetPoints(myPoints)
            myUnGrid.SetCells(vtk.VTK_TRIANGLE, myCells)
    
                
        if Tetra == True:
            #set pts
            for coords in self.node:
                myPoints.InsertNextPoint((coords[0], coords[1], coords[2]))
            
            #set Cells    
            for coords in self.elems:
                myCells.InsertNextCell(4)
                myCells.InsertCellPoint(coords[0])
                myCells.InsertCellPoint(coords[1])
                myCells.InsertCellPoint(coords[2])
                myCells.InsertCellPoint(coords[3])
                
            myUnGrid.SetPoints(myPoints)
            myUnGrid.SetCells(vtk.VTK_TETRA, myCells)
        
        #scalar
        myScalars = vtk.vtkFloatArray()
        myScalars.SetNumberOfComponents(1)
        
        for i in range(0, len(vector)):
            myScalars.InsertNextTuple1(vector[i])
        
        myUnGrid.GetPointData().SetScalars(myScalars)
        #mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInput(myUnGrid)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarRange(np.min(vector), np.max(vector))

        #actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        #renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor)

        #render window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300,300)

        renWin.Render()
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renWin)
        interactor.Initialize()
        interactor.Start()
        
    
    
    def ReadFixedTempNodes(self, fileName):
        file = open(fileName, 'r')
        
        K = int(file.readline())
        tempNodesArray = []
        tempArray=[]
        
        for i in range(0,K):
            #Read Temp
            tempInfo = file.readline().split()
            tempArray.append(float(tempInfo[1]))
            nodeInfo = file.readline().split()
            tempNodesArray.append(nodeInfo)
        
        for i in range(0,len(tempNodesArray)):
            temps =[]
            for j in tempNodesArray[i]:
                temps.append(int(j))
            tempNodesArray[i] = temps
            
        self.fixed_temps = tempArray
        self.fixed_temp_nodes = tempNodesArray
        
    
    def ComputeStiffnessMatrix(self, readings):
        X = spmatrix.ll_mat(len(self.node),len(self.node))
        
        #Keeping track of index for conductivity  
        conductIndex = 0
        
        for coords in self.elems:
            
            n1 = coords[0]
            n2 = coords[1]
            n3 = coords[2]
            
            n = [n1, n2, n3]
            
            x1 = self.node[n1][0]
            y1 = self.node[n1][1]
            x2 = self.node[n2][0]
            y2 = self.node[n2][1]
            x3 = self.node[n3][0]
            y3 = self.node[n3][1]
            
            a1 = y2-y3
            a2 = y3-y1
            a3 = y1-y2
            
            b1 = x3-x2
            b2 = x1-x3
            b3 = x2-x1
            
            Area = .5*((x2*y3)-
                          (x3*y2)+
                          (x3*y1)-
                          (x1*y3)+
                          (x1*y2)-
                          (x2*y1))
            
            gx1 = a1/(2*Area)
            gx2 = a2/(2*Area)
            gx3 = a3/(2*Area)
            
            gy1 = b1/(2*Area)
            gy2 = b2/(2*Area)
            gy3 = b3/(2*Area)
            
            gx = [gx1, gx2, gx3]
            gy = [gy1, gy2, gy3]
            
            for i in range(0,3):
                i_global = n[i]
                for j in range(0,3):
                    j_global = n[j]
                       
                    X[i_global, j_global]+= float((gx[i]*gx[j] + gy[i]*gy[j])*Area*readings[conductIndex])
            
            conductIndex += 1
        print X[0:5,0:5] 
        return X
        
    def SolveSetTempsProblem(self, conductVector):
        Z = self.ComputeStiffnessMatrix(conductVector)      
        f = np.zeros((len(self.node)), dtype=np.double)
         
        for n in self.fixed_temp_nodes[0]:
            Z[n,:]=0.0
            Z[n,n]=1.0
            f[n]=self.fixed_temps[0]
       
        for n in self.fixed_temp_nodes[1]:
            Z[n,:]=0.0
            Z[n,n]=1.0
            f[n]=self.fixed_temps[1]
        
        w = np.zeros((len(self.node)), dtype=np.double)
        LU = superlu.factorize(Z.to_csr())
        LU.solve(f, w)
        
        return w
                    
a = MyMesh()
a.ReadElements("2D_holes_mesh_elems.txt")
a.ReadNodes("2D_holes_mesh_nodes.txt")
a.ReadFixedTempNodes('2D_holes_mesh_fixed_temp_nodes.txt')

conduct = np.ones((len(a.elems),1), dtype=np.double)
Z = a.ComputeStiffnessMatrix(conduct)
w = a.SolveSetTempsProblem(conduct)
a.DisplayNodalScalars(w)


