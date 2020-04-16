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

import static org.jvmpy.python.Python.print;
import static org.jvmtorch.JvmTorch.torch;

import java.util.logging.LogManager;

// Spring imports
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;

/**
 * Based on PyTorch tutorial at:
 * https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
 *
 * This tutorial and the JvmTorch library aims for a Python-esque style of code,
 * deliberately deviating from standard Java code conventions.
 *
 */
public class TensorTutorial implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(TensorTutorial.class, args);
	}

	static {
		// Optional - quieten Logging for JBlas
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();
	}

	@Override
	public void run(String... args)  {

		// Construct a 5x3 matrix, uninitialized:
		
		var x = torch.empty(5, 3);
		print(x);
		
		// Construct a randomly initialized matrix:
		
		x = torch.rand(5, 3);
		print(x);
		
		// Construct a matrix filled zeros 
		x = torch.zeros(5, 3);
		print(x);
		
		// Get its Size
		
		print(x.size());
		
		
		// Construct a tensor directly from data:
		
		x = torch.tensor(new float[] {5.5f, 3f });
		print(x);
		
		
		// There are multiple syntaxes for operations. 
		// In the following example, we will take a look at the addition operation.
		
		// Addition
		
		x = torch.randn(5, 3);
		var y = torch.rand(5, 3);
		
		// Addition, syntax 1
		print(torch.add(x, y));
		
		/// Addition, syntax 2
		print(x.add(y));
		
		// Addition in-place
		print(x.add_(y));
	}

}
