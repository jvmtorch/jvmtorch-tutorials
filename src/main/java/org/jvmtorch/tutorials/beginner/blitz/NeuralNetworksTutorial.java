/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.jvmtorch.tutorials.beginner.blitz;

import static org.jvmpy.python.Python.True;
import static org.jvmpy.python.Python.len;
import static org.jvmpy.python.Python.list;
import static org.jvmpy.python.Python.print;
import static org.jvmpy.python.Python.tuple;
import static org.jvmtorch.JvmTorch.nn;
import static org.jvmtorch.JvmTorch.optim;
import static org.jvmtorch.JvmTorch.torch;

import java.util.logging.LogManager;

import org.jvmtorch.nn.Conv2d;
import org.jvmtorch.nn.Linear;
import org.jvmtorch.nn.Module;
import org.jvmtorch.nn.NN;
import org.jvmtorch.torch.Tensor;
// Spring imports
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Based on PyTorch tutorial at:
 * https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
 *
 * This tutorial and the JvmTorch library aims for a Python-esque style of code,
 * deliberately deviating from standard Java code conventions.
 *
 */
@SpringBootApplication
public class NeuralNetworksTutorial implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(NeuralNetworksTutorial.class, args);
	}

	static {
		// Optional - quieten Logging for JBlas
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();
	}

	// Let’s define a network:

	/**
	 * The following imports have been added to the top of this class to support the
	 * following code section.
	 * 
	 * import static org.jvmpy.python.Python.*;
	 * import org.jvmtorch.nn.modules.Module;
	 * import org.jvmtorch.torch.Tensor;
	 * import org.jvmtorch.torch.TensorOperations;
	 * 
	 **/
	public static class Net extends Module<Net> {

		protected Conv2d<?> conv1;
		protected Conv2d<?> conv2;
		protected Linear<?> fc1;
		protected Linear<?> fc2; // Implementing ModuleAttributes provides a num_flat_feature() method
		protected Linear<?> fc3;

		public Net(NN nn) {
			super(nn);
			// # 1 input image channel, 6 output channels, 3x3 square convolution
			// # kernel
			self.conv1 = nn.Conv2d(1, 6, 5).alias_("conv1");
			self.conv2 = nn.Conv2d(6, 16, 5).alias_("conv2");;
			// an affine operation: y = Wx + b
			self.fc1 = nn.Linear(16 * 5 * 5, 120).alias_("fc1");; // # 5*5 from image dimension
			self.fc2 = nn.Linear(120, 84).alias_("fc2");;
			self.fc3 = nn.Linear(84, 10).alias_("fc3");;
		}

		@Override
		public Tensor forward(Tensor x) {

			// Max pooling over a (2, 2) window
			x = F.max_pool2d(self.conv1.apply(x), Size(2, 2));
			// # If the size is a square you can only specify a single number
			x = F.max_pool2d(self.conv2.apply(x), 2);
			//x = x.view(-1, self.num_flat_features(x));
			x = self.fc1.apply(x);
			x = self.fc2.apply(x);
			x = self.fc3.apply(x);
			return x;
		}

		protected int num_flat_features(Tensor x) {
			// int size = x.size()[1:] # all dimensions except the batch dimension
			int[] size = x.size().dimensions();
			var num_features = 1;
			for (var s : size)
				num_features *= s;
			return num_features;
		}

		@Override
		protected Net self() {
			return this;
		}
	}

	@Override
	public void run(String... args)  {

		// Let’s create our network:
		var net = new Net(nn);
		print(net);

		// You just have to define the forward function, and the backward function
		// (where gradients are computed) is automatically defined for you using autograd.
		//
		// You can use any of the Tensor operations in the forward function.

		// The learnable parameters of a model are returned by net.parameters()

		var params = list(net.parameters());
		print(len(params));
		print(params[0]);

		print(params[0].size()); // # conv1's .weight

		// Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32.
		// To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.

		/*
		 * The following imports have been added to the top of this class to support the
		 * following code section.
		 *
		 * import static org.jvmtorch.JvmTorch.*;
		 */

		var input = torch.randn(torch.Size(torch.Size(1), torch.Size(1, 32, 32))).requires_grad_(True).names_(tuple("example", "input_depth", "input_height", "input_width"));
				
		var out = net.apply(input);
		print(out);

		// Zero the gradient buffers of all parameters and backprops with random gradients:
		net.zero_grad();
		out.backward(torch.randn(torch.Size(1, 10).names_(tuple("example", "feature"))));

		// A loss function takes the (output, target) pair of inputs, and computes a
		// value that estimates how far away the output is from the target.

		// There are several different loss functions under the nn package . 
		// A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.

		// For example:
		
		var output = net.apply(input);
		var target = torch.randn(torch.Size(1, 10)).names_(tuple("example", "feature")); // # a dummy target, for example
		//target = target.view(1, -1); // # make it the same shape as output
		var criterion = nn.MSELoss();

		var loss = criterion.apply(output, target);
		print(loss);


		// Now, if you follow loss in the backward direction, using its .grad_fn
		// attribute, you will see a graph of computations that looks like this:

		/*
		 * input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view
		 * 		-> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss
		 */

		// So, when we call loss.backward(), the whole graph is differentiated w.r.t.
		// the loss, and all Tensors in the graph that has requires_grad=True will have their .grad Tensor
		// accumulated with the gradient.

		// For illustration, let us follow a few steps backward:

		print(loss.grad_fn()); // # MSELoss
		print(loss.grad_fn().next_functions().get(0, 0)); // # Linear
		print(loss.grad_fn().next_functions().get(0).get(0).next_functions().get(0).get(0)); // # ReLU

		// To backpropagate the error all we have to do is to loss.backward(). 
		// You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.

		// Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.

		net.zero_grad(); // # zeroes the gradient buffers of all parameters

		print("conv1.bias.grad before backward");

		print(net.fc1.weight().grad());

		loss.backward();

		print("conv1.bias.grad after backward");
		print(net.fc1.weight().grad());
		
		// However, as you use neural networks, you want to use various different update
		// rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. 
		// To enable this, we built a small package: optim that implements all these methods. 
		// Using it is very simple:

		// # create your optimizer
		var optimizer = optim.SGD(net.parameters(), 0.01f);

		// # in your training loop:
		optimizer.zero_grad(); // # zero the gradient buffers
		output = net.apply(input);
		loss = criterion.apply(output, target);
		loss.backward();
		optimizer.step(); // # Does the update
	}

}
