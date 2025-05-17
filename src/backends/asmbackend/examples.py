"""
Example assembly code snippets for common operations on different architectures.

This module provides example assembly code templates for various
operations on ARM and x86_64 architectures, which can be used
as building blocks for more complex computational kernels.
"""

import platform
from typing import Dict, List, Optional, Tuple


class ARMExamples:
    """Example assembly code snippets for ARM architecture."""
    
    @staticmethod
    def vector_add_float32(input1: str, input2: str, output: str, length: int) -> str:
        """
        Generate ARM assembly for float32 vector addition.
        
        Args:
            input1: Input vector 1 name (pointer to float32 array)
            input2: Input vector 2 name (pointer to float32 array)
            output: Output vector name (pointer to float32 array)
            length: Vector length (number of float32 elements)
        
        Returns:
            ARM assembly code as string
        """
        # Determine if we can use SVE instructions
        use_sve = False  # Set to True if SVE detection is implemented
        
        if use_sve:
            # SVE implementation
            return f"""
            .global vector_add_float32
            .type vector_add_float32, %function
            
            // Function: vector_add_float32(float* {input1}, float* {input2}, float* {output}, int {length})
            // Arguments:
            //   x0 = {input1} - pointer to first input vector
            //   x1 = {input2} - pointer to second input vector
            //   x2 = {output} - pointer to output vector
            //   w3 = {length} - number of elements
            
            vector_add_float32:
                // Save registers
                stp x29, x30, [sp, #-16]!
                mov x29, sp
                
                // Initialize SVE predicate register for vector loop
                whilelt p0.s, wzr, w3
                
            1:  // Vector loop
                // Load vectors from {input1} and {input2}
                ld1w {{z0.s}}, p0/z, [x0]
                ld1w {{z1.s}}, p0/z, [x1]
                
                // Add vectors
                fadd z2.s, z0.s, z1.s
                
                // Store result to {output}
                st1w {{z2.s}}, p0, [x2]
                
                // Increment pointers by vector length
                incw x0, p0, #1
                incw x1, p0, #1  
                incw x2, p0, #1
                
                // Continue loop if there are more elements
                whilelt p0.s, w3, w3
                b.first 1b
                
                // Restore registers and return
                ldp x29, x30, [sp], #16
                ret
            """
        else:
            # NEON implementation
            return f"""
            .global vector_add_float32
            .type vector_add_float32, %function
            
            // Function: vector_add_float32(float* {input1}, float* {input2}, float* {output}, int {length})
            // Arguments:
            //   x0 = {input1} - pointer to first input vector
            //   x1 = {input2} - pointer to second input vector
            //   x2 = {output} - pointer to output vector
            //   w3 = {length} - number of elements
            
            vector_add_float32:
                // Save registers
                stp x29, x30, [sp, #-16]!
                mov x29, sp
                
                // Calculate number of 4-element chunks
                lsr w4, w3, #2  // w4 = length / 4
                
                // Handle 4 elements at a time with NEON
                cbz w4, 2f      // Skip if no complete chunks
                
            1:  // Main loop - process 4 elements at a time
                // Load 4 elements from each input
                ld1 {{v0.4s}}, [x0], #16
                ld1 {{v1.4s}}, [x1], #16
                
                // Add vectors
                fadd v2.4s, v0.4s, v1.4s
                
                // Store result
                st1 {{v2.4s}}, [x2], #16
                
                // Decrement counter and loop if not done
                subs w4, w4, #1
                b.ne 1b
                
            2:  // Handle remaining elements one at a time
                // Calculate remaining elements
                and w4, w3, #3  // w4 = length % 4
                cbz w4, 3f      // Skip if no remaining elements
                
            4:  // Remainder loop - one element at a time
                // Load individual elements
                ldr s0, [x0], #4
                ldr s1, [x1], #4
                
                // Add elements
                fadd s2, s0, s1
                
                // Store result
                str s2, [x2], #4
                
                // Decrement counter and loop if not done
                subs w4, w4, #1
                b.ne 4b
                
            3:  // Exit
                // Restore registers and return
                ldp x29, x30, [sp], #16
                ret
            """
    
    @staticmethod
    def matrix_multiply_float32(matrix_a: str, matrix_b: str, matrix_c: str, 
                               m: int, n: int, k: int) -> str:
        """
        Generate ARM assembly for float32 matrix multiplication (C = A * B).
        
        Args:
            matrix_a: Input matrix A name (pointer to m x k float32 array)
            matrix_b: Input matrix B name (pointer to k x n float32 array)
            matrix_c: Output matrix C name (pointer to m x n float32 array)
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A and rows in B
        
        Returns:
            ARM assembly code as string
        """
        # NEON implementation for a simple matrix multiplication
        return f"""
        .global matrix_multiply_float32
        .type matrix_multiply_float32, %function
        
        // Function: matrix_multiply_float32(float* {matrix_a}, float* {matrix_b}, float* {matrix_c}, int {m}, int {n}, int {k})
        // Arguments:
        //   x0 = {matrix_a} - pointer to matrix A (m x k)
        //   x1 = {matrix_b} - pointer to matrix B (k x n)
        //   x2 = {matrix_c} - pointer to matrix C (m x n)
        //   w3 = {m} - number of rows in A and C
        //   w4 = {n} - number of columns in B and C
        //   w5 = {k} - number of columns in A and rows in B
        
        matrix_multiply_float32:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            
            // Save additional callee-saved registers
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            
            // Initialize loop variables
            mov x19, #0         // i = 0 (row index for A and C)
            
        1:  // Loop over rows of A
            cmp x19, x3         // Compare i with m
            b.ge 7f             // Exit if i >= m
            
            mov x20, #0         // j = 0 (column index for B and C)
            
        2:  // Loop over columns of B
            cmp x20, x4         // Compare j with n
            b.ge 6f             // Go to next row of A if j >= n
            
            // Calculate offset for C[i,j]
            mul x23, x19, x4    // i * n
            add x23, x23, x20   // i * n + j
            lsl x23, x23, #2    // (i * n + j) * 4 bytes
            add x23, x2, x23    // &C[i,j]
            
            // Initialize sum = 0
            fmov s0, wzr
            
            mov x21, #0         // l = 0 (index for dot product)
            
        3:  // Loop for dot product
            cmp x21, x5         // Compare l with k
            b.ge 5f             // Done with dot product if l >= k
            
            // Calculate offset for A[i,l]
            mul x24, x19, x5    // i * k
            add x24, x24, x21   // i * k + l
            lsl x24, x24, #2    // (i * k + l) * 4 bytes
            add x24, x0, x24    // &A[i,l]
            
            // Calculate offset for B[l,j]
            mul x22, x21, x4    // l * n
            add x22, x22, x20   // l * n + j
            lsl x22, x22, #2    // (l * n + j) * 4 bytes
            add x22, x1, x22    // &B[l,j]
            
            // Load values and multiply-accumulate
            ldr s1, [x24]       // A[i,l]
            ldr s2, [x22]       // B[l,j]
            fmadd s0, s1, s2, s0 // sum += A[i,l] * B[l,j]
            
            add x21, x21, #1    // l++
            b 3b                // Continue dot product
            
        5:  // Store result C[i,j] = sum
            str s0, [x23]
            
            add x20, x20, #1    // j++
            b 2b                // Continue with next column
            
        6:  // Next row of A
            add x19, x19, #1    // i++
            b 1b                // Continue with next row
            
        7:  // Exit
            // Restore callee-saved registers
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            
            // Restore frame pointer and link register
            ldp x29, x30, [sp], #16
            ret
        """


class X86Examples:
    """Example assembly code snippets for x86_64 architecture."""
    
    @staticmethod
    def vector_add_float32(input1: str, input2: str, output: str, length: int, syntax: str = "att") -> str:
        """
        Generate x86_64 assembly for float32 vector addition.
        
        Args:
            input1: Input vector 1 name (pointer to float32 array)
            input2: Input vector 2 name (pointer to float32 array)
            output: Output vector name (pointer to float32 array)
            length: Vector length (number of float32 elements)
            syntax: Assembly syntax ("att" or "intel")
        
        Returns:
            x86_64 assembly code as string
        """
        if syntax == "intel":
            # Intel syntax
            return f"""
            .intel_syntax noprefix
            .global vector_add_float32
            .type vector_add_float32, @function
            
            # Function: vector_add_float32(float* {input1}, float* {input2}, float* {output}, int {length})
            # Arguments:
            #   rdi = {input1} - pointer to first input vector
            #   rsi = {input2} - pointer to second input vector
            #   rdx = {output} - pointer to output vector
            #   ecx = {length} - number of elements
            
            vector_add_float32:
                # Save registers
                push rbp
                mov rbp, rsp
                
                # Calculate number of 8-element chunks
                mov r8d, ecx
                shr r8d, 3           # r8d = length / 8
                
                # Handle 8 elements at a time with AVX
                test r8d, r8d
                jz avx_done          # Skip if no complete chunks
                
            avx_loop:
                # Load 8 elements (32 bytes) from each input
                vmovups ymm0, [rdi]
                vmovups ymm1, [rsi]
                
                # Add vectors
                vaddps ymm2, ymm0, ymm1
                
                # Store result
                vmovups [rdx], ymm2
                
                # Increment pointers by 32 bytes (8 floats)
                add rdi, 32
                add rsi, 32
                add rdx, 32
                
                # Decrement counter and loop if not done
                dec r8d
                jnz avx_loop
                
            avx_done:
                # Calculate remaining elements
                and ecx, 7           # ecx = length % 8
                test ecx, ecx
                jz done              # Skip if no remaining elements
                
            scalar_loop:
                # Handle one element at a time
                movss xmm0, [rdi]
                movss xmm1, [rsi]
                addss xmm0, xmm1
                movss [rdx], xmm0
                
                # Increment pointers by 4 bytes (1 float)
                add rdi, 4
                add rsi, 4
                add rdx, 4
                
                # Decrement counter and loop if not done
                dec ecx
                jnz scalar_loop
                
            done:
                # Clean up and return
                vzeroupper           # Clean up AVX state
                pop rbp
                ret
            """
        else:
            # AT&T syntax
            return f"""
            .att_syntax
            .global vector_add_float32
            .type vector_add_float32, @function
            
            # Function: vector_add_float32(float* {input1}, float* {input2}, float* {output}, int {length})
            # Arguments:
            #   %rdi = {input1} - pointer to first input vector
            #   %rsi = {input2} - pointer to second input vector
            #   %rdx = {output} - pointer to output vector
            #   %ecx = {length} - number of elements
            
            vector_add_float32:
                # Save registers
                pushq %rbp
                movq %rsp, %rbp
                
                # Calculate number of 8-element chunks
                movl %ecx, %r8d
                shrl $3, %r8d        # r8d = length / 8
                
                # Handle 8 elements at a time with AVX
                testl %r8d, %r8d
                jz avx_done          # Skip if no complete chunks
                
            avx_loop:
                # Load 8 elements (32 bytes) from each input
                vmovups (%rdi), %ymm0
                vmovups (%rsi), %ymm1
                
                # Add vectors
                vaddps %ymm1, %ymm0, %ymm2
                
                # Store result
                vmovups %ymm2, (%rdx)
                
                # Increment pointers by 32 bytes (8 floats)
                addq $32, %rdi
                addq $32, %rsi
                addq $32, %rdx
                
                # Decrement counter and loop if not done
                decl %r8d
                jnz avx_loop
                
            avx_done:
                # Calculate remaining elements
                andl $7, %ecx        # ecx = length % 8
                testl %ecx, %ecx
                jz done              # Skip if no remaining elements
                
            scalar_loop:
                # Handle one element at a time
                movss (%rdi), %xmm0
                movss (%rsi), %xmm1
                addss %xmm1, %xmm0
                movss %xmm0, (%rdx)
                
                # Increment pointers by 4 bytes (1 float)
                addq $4, %rdi
                addq $4, %rsi
                addq $4, %rdx
                
                # Decrement counter and loop if not done
                decl %ecx
                jnz scalar_loop
                
            done:
                # Clean up and return
                vzeroupper           # Clean up AVX state
                popq %rbp
                ret
            """
    
    @staticmethod
    def matrix_multiply_float32(matrix_a: str, matrix_b: str, matrix_c: str, 
                               m: int, n: int, k: int, syntax: str = "att") -> str:
        """
        Generate x86_64 assembly for float32 matrix multiplication (C = A * B).
        
        Args:
            matrix_a: Input matrix A name (pointer to m x k float32 array)
            matrix_b: Input matrix B name (pointer to k x n float32 array)
            matrix_c: Output matrix C name (pointer to m x n float32 array)
            m: Number of rows in A and C
            n: Number of columns in B and C
            k: Number of columns in A and rows in B
            syntax: Assembly syntax ("att" or "intel")
        
        Returns:
            x86_64 assembly code as string
        """
        if syntax == "intel":
            # Intel syntax
            return f"""
            .intel_syntax noprefix
            .global matrix_multiply_float32
            .type matrix_multiply_float32, @function
            
            # Function: matrix_multiply_float32(float* {matrix_a}, float* {matrix_b}, float* {matrix_c}, int {m}, int {n}, int {k})
            # Arguments:
            #   rdi = {matrix_a} - pointer to matrix A (m x k)
            #   rsi = {matrix_b} - pointer to matrix B (k x n)
            #   rdx = {matrix_c} - pointer to matrix C (m x n)
            #   ecx = {m} - number of rows in A and C
            #   r8d = {n} - number of columns in B and C
            #   r9d = {k} - number of columns in A and rows in B
            
            matrix_multiply_float32:
                # Save registers
                push rbp
                mov rbp, rsp
                push r12
                push r13
                push r14
                push r15
                
                # Save parameters to callee-saved registers
                mov r12, rdi        # A
                mov r13, rsi        # B
                mov r14, rdx        # C
                mov r15d, r9d       # k
                
                xor r10, r10        # i = 0 (row index for A and C)
                
            row_loop:
                cmp r10d, ecx       # Compare i with m
                jge done            # Exit if i >= m
                
                xor r11, r11        # j = 0 (column index for B and C)
                
            col_loop:
                cmp r11d, r8d       # Compare j with n
                jge next_row        # Go to next row of A if j >= n
                
                # Calculate offset for C[i,j]
                mov rax, r10
                mul r8              # rax = i * n
                add rax, r11        # rax = i * n + j
                shl rax, 2          # rax = (i * n + j) * 4 bytes
                lea rdi, [r14 + rax] # rdi = &C[i,j]
                
                # Initialize sum = 0
                vxorps xmm0, xmm0, xmm0
                
                xor rdx, rdx        # l = 0 (index for dot product)
                
            dot_loop:
                cmp edx, r15d       # Compare l with k
                jge store_result    # Done with dot product if l >= k
                
                # Calculate offset for A[i,l]
                mov rax, r10
                mul r15             # rax = i * k
                add rax, rdx        # rax = i * k + l
                shl rax, 2          # rax = (i * k + l) * 4 bytes
                
                # Load A[i,l]
                movss xmm1, [r12 + rax]
                
                # Calculate offset for B[l,j]
                mov rax, rdx
                mul r8              # rax = l * n
                add rax, r11        # rax = l * n + j
                shl rax, 2          # rax = (l * n + j) * 4 bytes
                
                # Load B[l,j] and multiply-accumulate
                movss xmm2, [r13 + rax]
                vfmadd231ss xmm0, xmm1, xmm2  # sum += A[i,l] * B[l,j]
                
                inc rdx             # l++
                jmp dot_loop
                
            store_result:
                # Store result C[i,j] = sum
                movss [rdi], xmm0
                
                inc r11             # j++
                jmp col_loop
                
            next_row:
                inc r10             # i++
                jmp row_loop
                
            done:
                # Restore callee-saved registers
                pop r15
                pop r14
                pop r13
                pop r12
                pop rbp
                ret
            """
        else:
            # AT&T syntax
            return f"""
            .att_syntax
            .global matrix_multiply_float32
            .type matrix_multiply_float32, @function
            
            # Function: matrix_multiply_float32(float* {matrix_a}, float* {matrix_b}, float* {matrix_c}, int {m}, int {n}, int {k})
            # Arguments:
            #   %rdi = {matrix_a} - pointer to matrix A (m x k)
            #   %rsi = {matrix_b} - pointer to matrix B (k x n)
            #   %rdx = {matrix_c} - pointer to matrix C (m x n)
            #   %ecx = {m} - number of rows in A and C
            #   %r8d = {n} - number of columns in B and C
            #   %r9d = {k} - number of columns in A and rows in B
            
            matrix_multiply_float32:
                # Save registers
                pushq %rbp
                movq %rsp, %rbp
                pushq %r12
                pushq %r13
                pushq %r14
                pushq %r15
                
                # Save parameters to callee-saved registers
                movq %rdi, %r12     # A
                movq %rsi, %r13     # B
                movq %rdx, %r14     # C
                movl %r9d, %r15d    # k
                
                xorl %r10d, %r10d   # i = 0 (row index for A and C)
                
            row_loop:
                cmpl %ecx, %r10d    # Compare i with m
                jge done            # Exit if i >= m
                
                xorl %r11d, %r11d   # j = 0 (column index for B and C)
                
            col_loop:
                cmpl %r8d, %r11d    # Compare j with n
                jge next_row        # Go to next row of A if j >= n
                
                # Calculate offset for C[i,j]
                movq %r10, %rax
                mulq %r8            # rax = i * n
                addq %r11, %rax     # rax = i * n + j
                shlq $2, %rax       # rax = (i * n + j) * 4 bytes
                leaq (%r14,%rax), %rdi # rdi = &C[i,j]
                
                # Initialize sum = 0
                vxorps %xmm0, %xmm0, %xmm0
                
                xorl %edx, %edx     # l = 0 (index for dot product)
                
            dot_loop:
                cmpl %r15d, %edx    # Compare l with k
                jge store_result    # Done with dot product if l >= k
                
                # Calculate offset for A[i,l]
                movq %r10, %rax
                mulq %r15           # rax = i * k
                addq %rdx, %rax     # rax = i * k + l
                shlq $2, %rax       # rax = (i * k + l) * 4 bytes
                
                # Load A[i,l]
                movss (%r12,%rax), %xmm1
                
                # Calculate offset for B[l,j]
                movq %rdx, %rax
                mulq %r8            # rax = l * n
                addq %r11, %rax     # rax = l * n + j
                shlq $2, %rax       # rax = (l * n + j) * 4 bytes
                
                # Load B[l,j] and multiply-accumulate
                movss (%r13,%rax), %xmm2
                vfmadd231ss %xmm2, %xmm1, %xmm0  # sum += A[i,l] * B[l,j]
                
                incq %rdx           # l++
                jmp dot_loop
                
            store_result:
                # Store result C[i,j] = sum
                movss %xmm0, (%rdi)
                
                incq %r11           # j++
                jmp col_loop
                
            next_row:
                incq %r10           # i++
                jmp row_loop
                
            done:
                # Restore callee-saved registers
                popq %r15
                popq %r14
                popq %r13
                popq %r12
                popq %rbp
                ret
            """


def get_examples(arch: str = None, syntax: str = "att"):
    """
    Get architecture-specific assembly examples.
    
    Args:
        arch: Target architecture (auto-detected if None)
        syntax: Assembly syntax style ("att" or "intel" for x86_64)
    
    Returns:
        Appropriate examples class for the architecture
    """
    if not arch:
        machine = platform.machine().lower()
        if machine in ("arm", "aarch64", "arm64"):
            arch = "arm"
        elif machine in ("x86_64", "amd64", "x64"):
            arch = "x86_64"
        else:
            raise ValueError(f"Unsupported architecture: {machine}")
    
    if arch in ("arm", "aarch64", "arm64"):
        return ARMExamples()
    elif arch in ("x86_64", "x86", "amd64"):
        return X86Examples()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")