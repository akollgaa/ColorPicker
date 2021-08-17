"""
This program uses numpy to create a Neural Network capable of predicting whether colors
look better on a black or white background. The "Training" button is used to train the 
algorithm. Click which ever option you think looks best, and continue for a minimun of 200 times
for accurate results shown on the counter. The "skip" buttons skips to a different color if both options
are suitable. The "Complete Training" button stops the training to show you how the algorithm preforms. Click
anywhere in the circle to the next color. The "Final-AI" button is the tested and "final" version of the algorithm.
Use File > Show weights to see each of the weights. File > Show percents to see the exact percentage of what the
algorithm thinks is correct. This was a big project that took a long time to fully understand all the variables.

Made by Adam Kollgaard
8/12/19

"""

import tkinter as tk
import numpy as np
from random import randint
root = tk.Tk()
root.title("Neural Network")
canvas = tk.Canvas(root, width=400, height=300)
canvas.grid(row=0, columnspan=4, pady=10, padx=10)
canvas.focus_set()

#This class is used to display all the intformation.
class display(object):
	def __init__(self):
		canvas.create_oval(25, 75, 175, 225, outline="black", fill="white", tags="White")
		canvas.create_oval(225, 75, 375, 225, outline="black", fill="black", tags="Black")
		self.color = "grey"
		self.whiteWord = canvas.create_text(100, 145, text="COLOR", fill=self.color, font=("Arial", 20, "bold"))
		self.blackWord = canvas.create_text(300, 145, text="COLOR", fill=self.color, font=("Arial", 20, "bold"))
		
		#These lines are for creating all the buttons and text used.
		statement = "I think it looks better on %s: %d" % ("Color", 0)
		self.textOutput = tk.Label(root, text=statement)
		self.textOutput.grid(row=1, columnspan=4, pady=5)
		
		self.mode = "Mode: None"
		self.modeText = tk.Label(root, text=self.mode)
		self.modeText.grid(row=2, columnspan=4)
		
		trainingButton = tk.Button(root, text="Training", command=self.training)
		trainingButton.grid(row=3, column=0, pady=5)
		
		skipButton = tk.Button(root, text="Skip", command=self.training)
		skipButton.grid(row=3, column=1, pady=5)
		
		completeButton = tk.Button(root, text="Complete Training", command= lambda: self.complete(False))
		completeButton.grid(row=3, column=2, pady=5)
		
		finalButton = tk.Button(root, text="Final AI", command=self.final)
		finalButton.grid(row=3, column=3, pady=5)
		
		self.counter = 0
		self.counterText = tk.Label(root, text=self.counter)
		self.counterText.grid(row=4, columnspan=4)
		
		self.complete(True)
	#This function is used during the training period.
	def training(self):
		Artifical.w1 = np.random.rand(3, 4)
		Artifical.w2 = np.random.rand(4, 2)
		self.complete(True)
		self.mode = "Mode: Training"
		self.modeText.config(text=self.mode)
		
	#This function creates new colors and displays them.
	#It also goes through the algorithm and tells you what it thinks is correct.
	def complete(self, option):
		if(option != 5):
			self.mode = "Mode: Complete Training"
			self.modeText.config(text=self.mode)
		if(option == False):
			return
		r = randint(1, 255)
		g = randint(1, 255)
		b = randint(1, 255)
		color = "#%02x%02x%02x" % (r, g, b)
		canvas.itemconfig(self.whiteWord, fill=color)
		canvas.itemconfig(self.blackWord, fill=color)
		Artifical.inputActivation(r, g, b)
		answer = Artifical.forwardPropagation()
		if(answer[0] > answer[1]):
			statement = "I think it looks better on %s: %.2f" % ("White", answer[0])
			self.textOutput.config(text=statement)
		elif(answer[0] < answer[1]):
			statement = "I think it looks better on %s: %.2f" % ("Black", answer[1])
			self.textOutput.config(text=statement)
		if(Artifical.showingPercent == True):
			statement = "White Percent: %.1f %%" % (answer[0] * 100)
			Artifical.whitePercent.config(text=statement)
			statement = "Black Percent: %.1f %%" % (answer[1] * 100)
			Artifical.blackPercent.config(text=statement)
	
	#This is final version of the code that has gone through
	#200 training examples.
	def final(self):
		self.mode = "Mode: Final-AI"
		self.counter = 0
		self.counterText.config(text=self.counter)
		self.modeText.config(text=self.mode)
		#These are the weights for each matrix of weights.
		Artifical.w1 = np.array([[0.80867908, -0.48975655, 0.55035326, 0.6510742], 
								[2.06904036, -3.22533999, 2.03513071, 2.91917689], 
								[0.78334796, -0.26620062, 0.68007978, 0.49523235]])
		Artifical.w2 = np.array([[-1.12644833, 0.66407203], 
								[3.19436854, -3.11362009], 
								[-1.03997443, 0.65660688], 
								[-1.13658554, 1.89454937]])
		self.complete(5)
	
	#This function is used for click function for the canvas.
	def click(self, event):
		if(175 > event.x > 25 and 225 > event.y > 75):
			if(self.mode == "Mode: Complete Training"):
				self.complete(True)
			elif(self.mode == "Mode: Training"):
				self.counter += 1
				self.counterText.config(text=self.counter)
				Artifical.backPropagation((1, 0))
				self.training()
			elif(self.mode == "Mode: Final-AI"):
				self.final()
		elif(375 > event.x > 225 and 225 > event.y > 75):
			if(self.mode == "Mode: Complete Training"):
				self.complete(True)
			elif(self.mode == "Mode: Training"):
				self.counter += 1
				self.counterText.config(text=self.counter)
				Artifical.backPropagation((0, 1))
				self.training()
			elif(self.mode == "Mode: Final-AI"):
				self.final()
	
#This Neural Network has three inputs Red, Green, and Blue which are the values of the color.
#It then normalizes the data which is brought to 4 Hidden layer nodes. 12 weights are used here.
#After calculated they move on to the final 2 output Nodes where the first one is white and the second is black.
#There are 8 weights used between the hidden and output layer. 20 weights and 9 nodes total.
#There are no Biases used in the network which was used for simplicity.
class NeuralNetwork(object):
		
	def __init__(self):
		self.showingWeight = False
		self.showingPercent = False
		self.w1 = np.random.rand(3, 4)
		self.w2 = np.random.rand(4, 2)	
	
	#This takes in inputs and normalizes the data between 0 and 1.
	#Later I recommend just dividing the input by 255 / the total possible value.
	def inputActivation(self, r, g, b):
		if(r == 128):
			r = 129
		if(g == 128):
			g = 129
		if(b == 128):
			b = 129
		r = (r-128)/128
		g = (g-128)/128
		b = (b-128)/128
		self.x = np.array([r, g, b])
	
	#This calculates the sigmoid of X.
	def sigmoidActivation(self, x):
		value = 1 / (1 + np.exp(-x))
		return value
	#This is the derivative of the sigmoid function.
	def sigmoidPrime(self, x):
		value = np.exp(-x) / ((1 + np.exp(-x)) ** 2)
		return value
	
	def forwardPropagation(self):
		self.z1 = self.x.dot(self.w1) #1 equation: multiply the inputs by the weights which creates 4 output values.
		self.a1 = self.sigmoidActivation(self.z1) # Take the sigmoid of each of the 4 values to put between 0 and 1.
		self.z2 = self.a1.dot(self.w2) # Mutliply those 4 values by the next set of weights to yeild two values.
		self.y_hat = self.sigmoidActivation(self.z2) # Take the sigmoid of the last value to have the final set of values.
		
		#This displays the weights if the window is open.
		if(self.showingWeight == True):
			self.showWeights(False)
			self.textBox.config(text=self.statement)
		
		return self.y_hat
	
	def backPropagation(self, y):
		#Here we backpropagate the second set of weights through gradient descent.
		delta3 = -(y - self.y_hat) * self.sigmoidPrime(self.z2)
		delta3 = np.array([[delta3[0]], [delta3[1]]]) # This takes delta3 and essientally transposes it.
		derivativeW2 = delta3 * self.a1 # Here we take delta 3 and multiply it by the activation value
		derivativeW2 = derivativeW2.T # This transposes the derivative matrix which makes it eaiser to use later.
		
		#Here we are backpropogating the first set of weights between the input and hidden layer.
		delta2 = np.dot(delta3.T, self.w2.T) * self.sigmoidPrime(self.z1) # Remember that it is okay to tranpose matrix and change where the end up.
		derivativeW1 = delta2.T * self.x # Also the * symbol and np.dot(A, B) can do different things depending of the A and B be sure to use the right one.
		derivativeW1 = derivativeW1.T
		
		self.learningRate = 1.5 # This is the size of steps that gradient descent takes.
		
		#Here we subtract the weights of the old values by a certain step size.
		self.w1 = self.w1 - self.learningRate * derivativeW1
		self.w2 = self.w2 - self.learningRate * derivativeW2
		
		#This displays the weights of the window is open.
		if(self.showingWeight == True):
			self.showWeights(False)
			self.textBox.config(text=self.statement)
		
		return "\n"
	
	#The following functions are used to create new windows and to show certain values.
	def showWeights(self, option):
		if(option == True):
			self.showingWeight = True
			self.newWindow = tk.Toplevel(root)
			self.newWindow.resizable(False, False)
		weight1 = ""
		weight2 = ""
		for x in range(0, 3):
			for y in range(0, 4):
				weight1 += "{} ".format(round(self.w1[x][y], 3))
			weight1 += "\n"
		for x in range(0, 4):
			for y in range(0, 2):
				weight2 += "{} ".format(round(self.w2[x][y], 3))
			weight2 += "\n"
		self.statement = "Weight one: {} \n Weight Two: {}".format(weight1, weight2)
		if(option == True):
			self.textBox = tk.Button(self.newWindow, text=self.statement)
			self.textBox.grid()
			self.newWindow.protocol("WM_DELETE_WINDOW", self.closeWeight)
	#These are just here to close the windows effectively.	
	def closeWeight(self):
		self.showingWeight
		self.newWindow.destroy()
	#This one is used to show the percentage window.
	def showPercent(self):
		self.showingPercent = True
		self.percentWindow = tk.Toplevel(root)
		self.percentWindow.resizable(False, False)
		statement = "White Percent: {}%".format(round(self.y_hat[0], 2) * 100)
		self.whitePercent = tk.Label(self.percentWindow, text=statement)
		self.whitePercent.grid(row=0)
		statement = "Black Percent: {}%".format(round(self.y_hat[1], 2) * 100)
		self.blackPercent = tk.Label(self.percentWindow, text=statement)
		self.blackPercent.grid(row=1)
		self.percentWindow.protocol("WM_DELETE_WINDOW", self.closePercent)
		
		
	def closePercent(self):
		self.showingPercent = False
		self.percentWindow.destroy()
		
		

Artifical = NeuralNetwork() # Artifical is the object of the NeuralNetwork.
view = display() # view the is object of the display which is used to show everything.

#This is where the File menu at the top is created.
menubar = tk.Menu(root)

filemenu = tk.Menu(menubar)
filemenu.add_command(label="Show Weights", command= lambda: Artifical.showWeights(True))
filemenu.add_command(label="Show Percents", command= Artifical.showPercent)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.destroy)

menubar.add_cascade(label="File", menu=filemenu)

canvas.bind("<Button-1>", view.click)

root.config(menu=menubar)
root.mainloop()