// ---------------------------------------------------------------------
//
// Copyright (C) 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// verify that we can create tasks both via lambdas and via std::bind
// expressions. this obviously requires C++11
//
// this is a small variation of the _12 test


#include "../tests.h"
#include <unistd.h>

#include <deal.II/base/thread_management.h>


// return a double, to make sure we correctly identify the return type
// of the expressions used in new_task(...)
double test (int i)
{
  deallog << "Task " << i << " starting..." << std::endl;
  sleep (1);
  deallog << "Task " << i << " finished!" << std::endl;

  return 3.141;
}




int main()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);

  Threads::TaskGroup<double> tg;

  // use variations of ways we can declare lambdas
  tg += Threads::new_task ([]() -> double { return test(1); });
  tg += Threads::new_task ([]() -> double { return (float)test(2); });

  tg.join_all ();

  deallog << "OK" << std::endl;

  deallog.detach ();
  logfile.close ();
  sort_file_contents ("output");
}
