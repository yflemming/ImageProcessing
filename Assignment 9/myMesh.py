'''
Created on Apr 10, 2014

@author: Yuon
'''
import numpy as np
import sys
import vtk 
from pysparse.sparse import spmatrix
from pysparse.direct import superlu
import matplotlib.pyplot as plt

class MyMesh:
    def __init__(self):
        
        self.node = []
        self.elems = []
        self.surf_faces = []
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
            print "Input vector length does not length of class.node"
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
        mapper.SetScalarRange(np.min(vector), np.max(vector))
        mapper.SetInput(myUnGrid)
        mapper.SetColorModeToMapScalars()
        
        #colorbar
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetTitle("Scalar Values")
        scalarBar.SetLookupTable(mapper.GetLookupTable() )
        scalarBar.SetOrientationToVertical()
                
        #position it
        pos = scalarBar.GetPositionCoordinate()
        pos.SetCoordinateSystemToNormalizedViewport()
        pos.SetValue(0.85,0.05)
        scalarBar.SetWidth(.1)
        scalarBar.SetHeight(.95)

        #actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        #renderer
        ren = vtk.vtkRenderer()
        ren.AddActor(actor)
        ren.AddActor(scalarBar)

        #render window
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(300,300)

        renWin.Render()
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(renWin)
        interactor.Initialize()
        interactor.Start()
    
    def DisplayNodalScalarsFirstTime(self, scalars):

        if (self.node != []) and (self.elems != []):
            self.myPoints = vtk.vtkPoints()
            self.myCells = vtk.vtkCellArray()
            self.myUnGrid = vtk.vtkUnstructuredGrid()
            
            if len(scalars) == len(self.node): # check that # of scalars matches the # of pts 
                
                self.myScalars = vtk.vtkFloatArray()
                self.myScalars.SetNumberOfComponents(1) # scalar vector

                for i in range(0,len(self.node)):
                    self.myScalars.InsertNextTuple1(scalars[i])
                    
                self.myUnGrid.GetPointData().SetScalars(self.myScalars)    
            
                if len(self.elems[0])==3: # triangular mesh
                    
                    # define geometry by inserting points into a vtkPoints object
                    for i in range(0, len(self.node)):
                        self.myPoints.InsertNextPoint((self.node[i][0],self.node[i][1],0.0))
                        
                    # define topology by inserting cells into a vtkCells object
                    for i in range(0, len(self.elems)):
                        self.myCells.InsertNextCell(3) 
                        self.myCells.InsertCellPoint(self.elems[i][0])
                        self.myCells.InsertCellPoint(self.elems[i][1])
                        self.myCells.InsertCellPoint(self.elems[i][2])                 
                                      
                    # associate the geometry and topology to the unstructured grid
                    self.myUnGrid.SetPoints(self.myPoints)
                    self.myUnGrid.SetCells(vtk.VTK_TRIANGLE, self.myCells)
                    
                elif len(self.elems[0])==4: # tetrahedral mesh
                
                    for i in range(0,len(self.node)):
                        self.myPoints.InsertNextPoint((self.node[i][0],self.node[i][1],self.node[i][2]))
                        
                    # define topology by inserting cells into a vtkCells object
                    for i in range(0, len(self.elems)):
                        self.myCells.InsertNextCell(4) 
                        self.myCells.InsertCellPoint(self.elems[i][0])
                        self.myCells.InsertCellPoint(self.elems[i][1])
                        self.myCells.InsertCellPoint(self.elems[i][2])
                        self.myCells.InsertCellPoint(self.elems[i][3])
                        
                        # associate the geometry and topology to the unstructured grid
                        self.myUnGrid.SetPoints(self.myPoints)
                        self.myUnGrid.SetCells(vtk.VTK_TETRA, self.myCells)
                    
                else:
                    
                    print "unrecognized mesh type"
                    return
    
                #mapper
                self.mapper = vtk.vtkDataSetMapper()
                self.mapper.SetInput(self.myUnGrid)
            
                
                self.mapper.SetScalarRange(np.min(scalars), np.max(scalars))

                #actor
                actor = vtk.vtkActor()
                actor.SetMapper(self.mapper)
                
                actor.GetProperty().SetRepresentationToSurface()
                
                #colorbar
                self.scalarBar = vtk.vtkScalarBarActor()
                self.scalarBar.SetTitle("Scalar Values")
                self.scalarBar.SetLookupTable(self.mapper.GetLookupTable() )
                self.scalarBar.SetOrientationToVertical()
                
                #position it
                pos = self.scalarBar.GetPositionCoordinate()
                pos.SetCoordinateSystemToNormalizedViewport()
                pos.SetValue(0.85,0.05)
                self.scalarBar.SetWidth(.1)
                self.scalarBar.SetHeight(.95)
                
                #renderer
                ren = vtk.vtkRenderer()
                ren.AddActor(actor)
                ren.AddActor(self.scalarBar)

                #render window
                self.renWin = vtk.vtkRenderWindow()
                self.renWin.AddRenderer(ren)
                self.renWin.SetSize(600,450)

                #interactor
                self.interactor = vtk.vtkRenderWindowInteractor()
                self.interactor.SetRenderWindow(self.renWin)
                
                # ask VTK to render once, in a non blocking fashion
                self.renWin.Render()                                
                    
            else: # the input scalars vector does not match the number of nodes in the mesh
                print "cannot display scalars as the length if the input vector does not match the number of nodes in the mesh"
                return    
    
    def DisplayNodalScalarsUpdate(self, scalars):
        
        # inside this method we can access self.myScalars and update the values, to display a new set of temperatures
        if len(scalars) == self.myScalars.GetNumberOfTuples():
            
            # cycle on self.myScalars writing the new temperatures
            for i in range(0,len(scalars)):
                self.myScalars.SetTuple1(i,scalars[i])
             
            # update the scalar range for display    
            self.mapper.SetScalarRange(np.min(scalars), np.max(scalars))
            
            # we have to tell the UnstructuredGrid that we have modified it (by modifying the internal scalars)
            # this will update certain internal structures and it is required, otherwise the updated values will not show
            self.myUnGrid.Modified() 
            
            # ask VTK to render once, in a non blocking fashion
            self.renWin.Render() 
            
    def DisplayNodalScalarsAndWait(self, scalars):
        
        # this method will display the nodal scalars and start the interactor
        # resulting in a blocking call and allowing the user to interact with the scene
        
        # we can rely on the method below to display the scalars once
        self.DisplayNodalScalarsUpdate(scalars)
       
        # then we start the interactor, this would be a blockng call, the program execution
        # will stop here until the user will close the window
        # use of the interactor will give the user a chance to manipulate objects on screen

        self.interactor.Initialize() 
        self.interactor.Start()
    
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
        Z = spmatrix.ll_mat(len(self.node),len(self.node))
        computeTetra = False
        computeTri = False
        
        #Used to keep track of the index for conductivity
        conductivityIndex = 0
        
        if len(self.elems[0])==3:
            computeTri = True
        elif len(self.elems[0])==4:
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
                       
                        Z[i_global, j_global]+= float((gx[i]*gx[j] + gy[i]*gy[j])*Area*thermalReadings[conductivityIndex])
            
                conductivityIndex += 1
        
        
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
                       
                        Z[i_global, j_global]+= float((gx[i]*gx[j] + gy[i]*gy[j] + gz[i]*gz[j])*Volume*thermalReadings[conductivityIndex])
            
            conductivityIndex += 1     
        return Z
        
    def SolveSetTempsProblem(self, conductVector, Tf, convection_coeff):
        Z = self.ComputeStiffnessMatrix(conductVector)
        Z = self.ComputeConvectionContributionsToSystemMatrix(Z, convection_coeff)               
        f = np.zeros(len(self.node), dtype=np.double)
                
        for n in self.fixed_temp_nodes[0]:
            Z[n,:]=0.0
            Z[n,n]=1.0
            f[n]=self.fixed_temps[0]
       
        f = mesh.ComputeConvectionContributionsToRHS(f, Tf, convection_coeff)       
        
        w = np.zeros((len(self.node)), dtype=np.double)
        LU = superlu.factorize(Z.to_csr())
        LU.solve(f, w)
        
        return w
    
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
        
        #Scalar
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
        
        # colorbar
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetTitle("Scalar Values")
        scalarBar.SetLookupTable(clipperMapper.GetLookupTable() )
        scalarBar.SetOrientationToVertical()
                
        # position it 
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
        
    def ComputeConvectionContributionsToSystemMatrix(self, Y, convection_coeff):
        
        #Used to keep track of the index for conductivity  
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
                        Y[i_global, j_global] += (1.0/6.0)*Area*convection_coeff[conductivityIndex][0]
                    elif i!=j:
                        Y[i_global, j_global] += (1.0/12.0)*Area*convection_coeff[conductivityIndex][0]                  
            conductivityIndex += 1
        return Y  
     
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

    def ComputeMassMatrix(self, capacityVector):
        C = spmatrix.ll_mat(len(self.node),len(self.node))
        capIndex = 0
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
                                                           
            Area = 0.5*(x2*y3-x3*y2+x3*y1-x1*y3+x1*y2-x2*y1)  
                           
            for i in range(0,3):
                i_global = n[i]
                for j in range(0,3):
                    j_global = n[j]
                    
                    C[i_global, j_global] += float(capacityVector[capIndex]*(1.0/6.0)*Area)
            
            capIndex += 1
        return C
        
    def SolveHeatTransient(self,theta,conductivity,capacity,init_temps,final_time,time_step):
      
        self.DisplayNodalScalarsFirstTime(init_temps)
        C = self.ComputeMassMatrix(capacity)
        K = self.ComputeStiffnessMatrix(conductivity)
        
        
        A = C.copy()
        B = C.copy()
        
        T = init_temps
                
        for i in range(len(self.node)):
            for j in range(len(self.node)):
                A[i,j] += K[i,j]*(theta * time_step)
                       
        for i in range(len(self.node)):
            for j in range(len(self.node)):
                B[i,j] -= K[i,j]*((1-theta) * time_step)             
        
        LU = superlu.factorize(A.to_csr())
        b = np.zeros((len(self.node)), dtype=np.double)
        
        n=0
        while n<final_time:
            print np.min(T)
            print np.max(T)        
            B.matvec(T, b)        
            LU.solve(b,T)
            n+=time_step

            conductivity[n] = conductivity[n-1]
            
            self.DisplayNodalScalarsUpdate(T)
        
        self.DisplayNodalScalarsAndWait(T)
        
mesh = MyMesh()
mesh.ReadNodes("square_mesh_2D_nodes.txt")
mesh.ReadElements("square_mesh_2D_elems.txt")

init_temps = np.zeros((len(mesh.node)), dtype=np.double)

for k in range(len(mesh.node)):
    if (mesh.node[k][0]<=1) and (mesh.node[k][0]>=-1) and (mesh.node[k][1]<=1) and (mesh.node[k][1]>=-1):
            init_temps[k]=(100)
    else:
        init_temps[k]=(25)


conduct = np.ones((len(mesh.elems),1), dtype=np.double)
SndConductivity = np.zeros((len(mesh.elems)), dtype=np.double)

for i in range(len(mesh.elems)):
    print mesh.node[mesh.elems[0][0]]
    x_bary = (mesh.node[mesh.elems[i][0]][0]+mesh.node[mesh.elems[i][1]][0]+mesh.node[mesh.elems[i][2]][0])/3
    if x_bary<0:
        SndConductivity[i] = 1.0
    elif x_bary>0:
        SndConductivity[i] = 0.2

convection_coeff = np.ones((len(mesh.surf_faces),1), dtype=np.double)
capacity = np.ones((len(mesh.elems),1), dtype=np.double)

#mesh.DisplayNodalScalars(init_temps)
#C = mesh.ComputeMassMatrix(capacity)

mesh.SolveHeatTransient(1, SndConductivity, capacity, init_temps, 5, 0.1)
