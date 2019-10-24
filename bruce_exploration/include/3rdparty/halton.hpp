/*
HALTON is a C++ library which computes elements of a Halton Quasi Monte Carlo (QMC) sequence using a simple interface by https://people.sc.fsu.edu/~jburkardt/cpp_src/halton/halton.html.
*/

/*
* Copyright (C) 2007-2011 John Tsiombikas <nuclear@member.fsf.org>
*
* Redistribution and use in source and binary forms, with or without
*     modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
* 3. The name of the author may not be used to endorse or promote products
* derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
*     WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
* EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
*     OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
*     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
*     IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
*/

double *halton(int i, int m);
double *halton_base(int i, int m, int b[]);
int halton_inverse(double r[], int m);
double *halton_sequence(int i1, int i2, int m);
int i4vec_sum(int n, int a[]);
int prime(int n);
double r8_mod(double x, double y);
void r8mat_print(int m, int n, double a[], std::string title);
void r8mat_print_some(int m, int n, double a[], int ilo, int jlo, int ihi,
                      int jhi, std::string title);
void timestamp();

