'''
Created on April 9, 2014

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
        
        #Scalar
        myScalars = vtk.vtkFloatArray()
        myScalars.SetNumberOfComponents(1)
        
        for i in range(0, len(vector)):
            myScalars.InsertNextTuple1(vector[i])
        
        myUnGrid.GetPointData().SetScalars(myScalars)
        #mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetScalarRange(np.min(vector), np.max(vector))
        mapper.SetInput(myUnGrid)
        mapper.SetColorModeToMapScalars()

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
        
        for n in range(0,len(tempNodesArray)):
            temps =[]
            for num in tempNodesArray[n]:
                temps.append(int(num))
            tempNodesArray[n] = temps
            
        self.fixed_temps = tempArray
        self.fixed_temp_nodes = tempNodesArray

    
    def ComputeStiffnessMatrix(self, thermalReadings):
        Y = spmatrix.ll_mat(len(self.node),len(self.node))
        computeTetra = False
        computeTri = False
        
        #Keeping track of index for conductivity  
        conductIndex = 0
        
        if len(self.node)==3:
            computeTri = True
        else:
            computeTetra = True
        
        if computeTri == True:
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
                       
                        Y[i_global, j_global]+= float((gx[i]*gx[j] + gy[i]*gy[j])*Area*thermalReadings[conductIndex])
            
            conductIndex += 1
        
        
        if computeTetra == True:
            for coords in self.elems:
                n1 = coords[0]
                n2 = coords[1]
                n3 = coords[2]
                n4 = coords[3]
            
                n = [n1, n2, n3, n4]
            
                x1 = self.node[n1][0]
                y1 = self.node[n1][1]
                z1 = self.node[n1][2]
                                
                x2 = self.node[n2][0]
                y2 = self.node[n2][1]
                z2 = self.node[n2][2]
                    
                x3 = self.node[n3][0]
                y3 = self.node[n3][1]
                z3 = self.node[n3][2]
                     
                x4 = self.node[n4][0]
                y4 = self.node[n4][1]
                z4 = self.node[n4][2]
                
            
                a1=-(y3*z4-z3*y4-y2*z4+z2*y4+y2*z3-z2*y3)
                a2= (y4*z1-z4*y1-y3*z1+z3*y1+y3*z4-z3*y4)
                a3=-(y1*z2-z1*y2-y4*z2+z4*y2+y4*z1-z4*y1)
                a4= (y2*z3-z2*y3-y1*z3+z1*y3+y1*z2-z1*y2)
               
                b1=-(x2*z4-x2*z3-x3*z4+x3*z2+x4*z3-x4*z2)
                b2= (x3*z1-x3*z4-x4*z1+x4*z3+x1*z4-x1*z3)
                b3=-(x4*z2-x4*z1-x1*z2+x1*z4+x2*z1-x2*z4)
                b4= (x1*z3-x1*z2-x2*z3+x2*z1+x3*z2-x3*z1)
               
                c1=-(x2*y3-x2*y4-x3*y2+x3*y4+x4*y2-x4*y3)
                c2= (x3*y4-x3*y1-x4*y3+x4*y1+x1*y3-x1*y4)
                c3=-(x4*y1-x4*y2-x1*y4+x1*y2+x2*y4-x2*y1)
                c4= (x1*y2-x1*y3-x2*y1+x2*y3+x3*y1-x3*y2)

                Volume = (1.0/6.0)*(x2*y3*z4-x2*z3*y4-x3*y2*z4+x3*z2*y4+x4*y2*z3-x4*z2*y3-x1*y3*z4+x1*z3*y4+x3*y1*z4-x3*z1*y4-x4*y1*z3+x4*z1*y3+x1*y2*z4-x1*z2*y4-x2*y1*z4+x2*z1*y4+x4*y1*z2-x4*z1*y2-x1*y2*z3+x1*z2*y3+x2*y1*z3-x2*z1*y3-x3*y1*z2+x3*z1*y2)
                
                                   
                gx1 = a1/(6*Volume)
                gx2 = a2/(6*Volume)
                gx3 = a3/(6*Volume)
                gx4 = a4/(6*Volume)
            
                gy1 = b1/(6*Volume)
                gy2 = b2/(6*Volume)
                gy3 = b3/(6*Volume)
                gy4 = b4/(6*Volume)
                
                gz1 = c1/(6*Volume)
                gz2 = c2/(6*Volume)
                gz3 = c3/(6*Volume)
                gz4 = c4/(6*Volume)
            
                gx = [gx1, gx2, gx3, gx4]
                gy = [gy1, gy2, gy3, gy4]
                gz = [gz1, gz2, gz3, gz4]
                
                for i in range(0,4):
                    i_global = n[i]
                    for j in range(0,4):
                        j_global = n[j]
                       
                        Y[i_global, j_global]+= float((gx[i]*gx[j] + gy[i]*gy[j] + gz[i]*gz[j])*Volume*thermalReadings[conductIndex])
            
            conductIndex += 1     
        return Y
        
    def SolveSetTempsProblem(self, conductVector, Tf, convection_coeff):
        Z = self.ComputeStiffnessMatrix(conductVector)
        Z = self.ComputeConvectionContributionsToSystemMatrix(Z, convection_coeff)
              
        f = np.zeros(len(self.node), dtype=np.double)
        
        
        for n in self.fixed_temp_nodes[0]:
            Z[n,:]=0.0
            Z[n,n]=1.0
            f[n]=self.fixed_temps[0]
       
        f = self.ComputeConvectionContributionsToRHS(f, Tf, convection_coeff)
        w = np.zeros((len(self.node)), dtype=np.double)
        LU = superlu.factorize(Z.to_csr())
        LU.solve(f, w)
        
        
        return w
    
    #scalar bar
    def DisplayNodalScalarsClip(self, vector):
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
        
        #clipping
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)
        plane.SetNormal(0, 1, 0)
        
        clipper = vtk.vtkClipDataSet()
        clipper.SetClipFunction(plane)
        clipper.SetInput(myUnGrid)
        
        clipperMapper = vtk.vtkDataSetMapper()
        clipperMapper.SetInputConnection(clipper.GetOutputPort())
        clipperMapper.SetScalarRange(np.min(vector), np.max(vector))
        
        clipperActor = vtk.vtkActor()
        clipperActor.SetMapper(clipperMapper)
        
        #colorbar
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetTitle("Scalar Values")
        scalarBar.SetLookupTable(clipperMapper.GetLookupTable() )
        scalarBar.SetOrientationToVertical()
                
        #position bar
        pos = scalarBar.GetPositionCoordinate()
        pos.SetCoordinateSystemToNormalizedViewport()
        pos.SetValue(0.85,0.05)
        scalarBar.SetWidth(.1)
        scalarBar.SetHeight(.95)
        
        ren = vtk.vtkRenderer()
        ren.AddActor(clipperActor)
        ren.AddActor(scalarBar)
        
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        renWin.SetSize(300,300)

        renWin.Render()
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renWin)
        interactor.Initialize()
        interactor.Start()
        
    def ReadSurfaceFaces(self, fileName):
        try:
            inFile = open(fileName, 'r')
        except IOError:
            print "IOError: File","\'"+fileName+"\'","was not located in given directory or path"
            sys.exit()
        
        dim = inFile.readline().split()
    
        surfFaceMatrix = []
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
            
            surfFaceMatrix.append(tempArray)
                            
        self.surf_faces = surfFaceMatrix
        
    def ComputeConvectionContributionsToSystemMatrix(self, Z, convection_coeff):
        
        #Used to keep track of the index to get the conductivity from  
        conductivityIndex = 0
        
        for coords in self.surf_faces:
            n1 = coords[0]
            n2 = coords[1]
            n3 = coords[2]
            
            n = [n1, n2, n3]
            
            x1 = self.node[n1][0]
            y1 = self.node[n1][1]
            z1 = self.node[n1][2]
                
            x2 = self.node[n2][0]
            y2 = self.node[n2][1]
            z2 = self.node[n2][2]
                
            x3 = self.node[n3][0]
            y3 = self.node[n3][1]
            z3 = self.node[n3][2]
                
            vtx1 = np.array([x1,y1,z1])
            vtx2 = np.array([x2,y2,z2])
            vtx3 = np.array([x3,y3,z3])

            Area = np.linalg.norm(np.cross(vtx2-vtx1,vtx3-vtx1))
                                    
            for i in range(0,3):
                i_global = n[i]
                for j in range(0,3):
                    j_global = n[j]
                                                           
                    if i==j:
                        Z[i_global, j_global] += (1.0/6.0)*Area*convection_coeff[conductivityIndex][0]
                    elif i!=j:
                        Z[i_global, j_global] += (1.0/12.0)*Area*convection_coeff[conductivityIndex][0]                  
            conductivityIndex += 1
        return Z 
     
    def ComputeConvectionContributionsToRHS(self,f, Tf, conv_coeff):
        conductivityIndex = 0
        for coords in self.surf_faces:
            n1 = coords[0]
            n2 = coords[1]
            n3 = coords[2]
            
            n = [n1, n2, n3]
            
            x1 = self.node[n1][0]
            y1 = self.node[n1][1]
            z1 = self.node[n1][2]
                
            x2 = self.node[n2][0]
            y2 = self.node[n2][1]
            z2 = self.node[n2][2]
                
            x3 = self.node[n3][0]
            y3 = self.node[n3][1]
            z3 = self.node[n3][2]
                
            vtx1 = np.array([x1,y1,z1])
            vtx2 = np.array([x2,y2,z2])
            vtx3 = np.array([x3,y3,z3])

            Area = np.linalg.norm(np.cross(vtx2-vtx1,vtx3-vtx1))
            
            for i in range(3):
                f[n[i]]+= conv_coeff[conductivityIndex][0]*Tf*(1.0/3.0)*Area
            
            conductivityIndex += 1
        return f
                     
mesh = MyMesh()
mesh.ReadNodes("sphere_3D_nodes.txt")
mesh.ReadElements("sphere_3D_elems.txt")
mesh.ReadSurfaceFaces("sphere_3D_surf_faces.txt")
mesh.ReadFixedTempNodes("sphere_3D_fixed_temp_nodes_2.txt")

num_nodes = len(mesh.node)
num_elems = len(mesh.elems)
num_faces = len(mesh.surf_faces)

conductivity = np.ones((len(mesh.elems),1), dtype=np.double)
convection_coeff = np.ones((len(mesh.surf_faces),1), dtype=np.double)
# convection_coeff *= 0.5

w = mesh.SolveSetTempsProblem(conductivity,10,convection_coeff)

print "Min Solution", w.min()
print "Max Solution", w.max

mesh.DisplayNodalScalarsClip(w)

print 'terminated'
