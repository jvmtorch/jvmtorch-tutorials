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

import static org.jvmpy.python.Python.*;

import static org.jvmtorch.JvmTorch.torch;

import java.util.logging.LogManager;

// Spring imports
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;

/**
 * Based on PyTorch tutorial at:
 * https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
 *
 * This tutorial and the JvmTorch library aims for a Python-esque style of code,
 * deliberately deviating from standard Java code conventions.
 *
 */
public class AutogradTutorial implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(AutogradTutorial.class, args);
	}

	static {
		// Optional - quieten Logging for JBlas
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();
	}

	@Override
	public void run(String... args)  {

		// Create a tensor and set requires_grad=True to track computation with it
		var x = torch.ones(2, 2).requires_grad_(True);
		print(x);
		
		// Do a tensor operation:
		
		var y = x.add(2);
		print(y);
		
		// y was created as a result of an operation, so it has a grad_fn.
		//print(y.grad_fn());  // No grad function in latest implementation.
		
		// Do more operations on y
		var z = y.mul(y).mul(3);
		var out = z.mean();

		print(z);
		print(out);


		
		// .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. 
		// The input flag defaults to False if not given.
		
		var a = torch.randn(2, 2);
		a = ((a.mul(3)).div(a.sub(1)));
		
		print(a.requires_grad());
		a.requires_grad_(True);
		print(a.requires_grad());
		var b = (a.mul(a)).sum();
		//print(b.grad_fn()); // No grad function in latest implementation.
		
		// Let’s backprop now. Because out contains a single scalar, out.backward() 
		// is equivalent to out.backward(torch.tensor(1.)).
		
		out.backward();
		
		// Print gradients d(out)/dx
		
		print(x.grad());
		
		
		// Now let’s take a look at an example of vector-Jacobian product:
		
		x = torch.randn(3).requires_grad_(True);

		y = x.mul(2);
		
		var y_norm_data = y.norm().getDataAsFloatArray();
	
		while (y_norm_data[0] < 1000 && y_norm_data[1] < 1000 && y_norm_data[2] < 1000) {
			y = y.mul(2);
			y_norm_data = y.norm().getDataAsFloatArray();
		}

		print(y);
		
		// Now in this case y is no longer a scalar. torch.autograd could not compute the full 
		// Jacobian directly, but if we just want the vector-Jacobian product, simply pass the vector 
		// to backward as argument:

		var v = torch.tensor(new float[] {0.1f, 1.0f, 0.0001f });
		y.backward(v);

		print(x.grad());

	}

}
