#Learning Invariant Representations with Local Transformations<br />
######Kihyuk Sohn and Honglak Lee, ICML 2012<br />
######for any question, please leave a message: kihyuk.sohn@gmail.com 

##CIFAR-10 database

1. get minFunc for learning classifier

    `$ source prep_minFunc.sh`

    0. download unconstrained optimization toolbox from the following website: http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html <br />
    0. we include the minFunc license form at the bottom of the page.

2. open matlab and run scripts.m

    `$ matlab`<br />
    `>> optgpu = 1; % 1 for gpu, 0 for cpu`<br />
    `>> gpuDevice(gpu_id); % if optgpu == 1, initialize gpu with proper gpu_id`<br />
    `>> scripts`<br />

    0. include scripts for learning and testing TI-RBMs with different types of transformations, such as translation, rotation, and scale variation.




=======================================================================================
The minFunc license is as follows:

The software on this webpage is distributed under the FreeBSD-style license below.

Although it is not required, I would also appreciate that any re-distribution of the
software contains a link to the original webpage.  For example, the webpage for the 
'minFunc' software is: http://www.di.ens.fr/~mschmidt/Software/minFunc.html

Copyright 2005-2012 Mark Schmidt. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=======================================================================================
