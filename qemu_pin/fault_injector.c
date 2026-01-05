/*
 * Profiler Pin Tool
 * This tool profiles the jacobi program to identify all unique instructions
 * and their locations for the fault injection campaign.
 */

#include <stdio.h>

#include "qemu-plugin.h"

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

// Define Address type.
typedef uint64_t Address_t;

// Has any fault been injected?
bool fault_injected = false;

// Target instruction information.
bool target_found = false;
Address_t target_address = 0;
char target_register_name[8];
size_t target_bit_position = 0;
size_t target_exec_count = 0;
size_t current_exec_count = 0;

// Output file name.
static const char* output_filename = "out/fault_result.out";
static FILE *output_file = NULL;

static struct qemu_plugin_register* GetRegisterHandle()
{
    // Get all available registers
    GArray *registers = qemu_plugin_get_registers();

    struct qemu_plugin_register* target_handle = NULL;
    
    for (int i = 0; i < registers->len; i++)
    {
        qemu_plugin_reg_descriptor *descriptor = &g_array_index(registers, qemu_plugin_reg_descriptor, i);
        
        if (strcmp(descriptor->name, target_register_name) == 0)
        {
            target_handle = descriptor->handle;
            break;
        }
    }
    
    g_array_free(registers, TRUE);

    return target_handle;
}

// Return TRUE iff the routine is valid for profiling. This filters out routines out of the program's main logic.
static bool IsValidRoutine(const char *symbol)
{
    if (!symbol) return true;

    // 1. Common Memory Allocators (often inlined or static)
    if (strstr(symbol, "malloc")) return false;
    if (strstr(symbol, "free")) return false;
    if (strstr(symbol, "calloc")) return false;
    if (strstr(symbol, "realloc")) return false;
    if (strstr(symbol, "_int_malloc")) return false;
    if (strstr(symbol, "sysmalloc")) return false;
    if (strstr(symbol, "mmap")) return false;
    if (strstr(symbol, "munmap")) return false;
    if (strstr(symbol, "munmap_chunk")) return false;
    if (strstr(symbol, "memrchr")) return false;
    if (strstr(symbol, "memset")) return false;
    if (strstr(symbol, "brk")) return false;
    if (strstr(symbol, "sbrk")) return false;

    // 2. Startup / Shutdown / C++ Runtime Boilerplate
    if (strcmp(symbol, "_start") == 0) return false; 
    if (strstr(symbol, "frame_dummy")) return false;
    if (strstr(symbol, "register_tm_clones")) return false;
    if (strstr(symbol, "deregister_tm_clones")) return false;
    
    // 3. Common Library Functions (if statically linked)
    if (strcmp(symbol, "printf") == 0) return false;
    if (strcmp(symbol, "puts") == 0) return false;
    if (strcmp(symbol, "exit") == 0) return false;
    if (strcmp(symbol, "abort") == 0) return false;

    // 4. Filter out compiler-generated symbols and libraries.
    if (strncmp(symbol, "__", 2) == 0) return false;
    if (strncmp(symbol, "_IO", 3) == 0) return false;
    if (strncmp(symbol, "_dl", 3) == 0) return false;
    if (strncmp(symbol, "_itoa", 5) == 0) return false;
    if (strcmp(symbol, "_exit") == 0) return false;

    // Miscellaneous known non-useful routines.
    if (strcmp(symbol, "strncmp") == 0) return false;
    if (strcmp(symbol, "strchrnul") == 0) return false;
    if (strcmp(symbol, "strtoq") == 0) return false;
    if (strcmp(symbol, "getenv") == 0) return false;
    if (strcmp(symbol, "tcache_init") == 0) return false;
    if (strcmp(symbol, "getenv") == 0) return false;
    if (strcmp(symbol, "get_cie_encoding") == 0) return false;
    if (strcmp(symbol, "call_fini") == 0) return false;
    if (strcmp(symbol, "get_pc_range") == 0) return false;
    if (strcmp(symbol, "version_lock_lock_exclusive") == 0) return false;
    if (strcmp(symbol, "version_lock_unlock_exclusive") == 0) return false;
    if (strcmp(symbol, "new_do_write") == 0) return false;
    if (strcmp(symbol, "read_encoded_value_with_base") == 0) return false;
    if (strcmp(symbol, "fstat64") == 0) return false;
    if (strcmp(symbol, "pthread_mutex_unlock") == 0) return false;
    if (strcmp(symbol, "release_registered_frames") == 0) return false;
    if (strncmp(symbol, "MAIN_", 5) == 0) return false;

    return true;
}

// Copy string and compact multiple spaces/tabs into single space in disassembly.
static void CompactSpacesCopy(char *src, char* dest)
{
    if (!src || !dest) return;
    
    bool space_seen = false;
    
    while (*src)
    {
        if (*src == ' ' || *src == '\t')
        {
            if (!space_seen)
            {
                *dest++ = ' ';
                space_seen = true;
            }
        }
        else
        {
            *dest++ = *src;
            space_seen = false;
        }

        src++;
    }

    *dest = '\0';
}

// Parse command-line arguments.
static bool ParseArguments(int argc, char **argv)
{
    bool seen_address = false;
    bool seen_bit = false;
    bool seen_count = false;

    target_address = 0;
    target_bit_position = 0;
    target_exec_count = 0;

    for (int i = 0; i < argc; i++)
    {
        char *opt = argv[i];

        if (strncmp(opt, "addr=", 5) == 0)
        {
            // strtoull parses decimal or hex (if 0x prefix exists) automatically with base 0
            target_address = (Address_t) strtoull(opt + 5, NULL, 0);
            seen_address = true;
        }
        else if (strncmp(opt, "bit=", 4) == 0)
        {
            target_bit_position = (size_t)strtoull(opt + 4, NULL, 0);
            seen_bit = true;
        }
        else if (strncmp(opt, "count=", 6) == 0)
        {
            target_exec_count = (size_t)strtoull(opt + 6, NULL, 0);
            seen_count = true;
        }
        else
        {
            fprintf(stderr, "[FAULT INJECTOR] Warning: Unknown argument '%s'\n", opt);
        }
    }

    if (!seen_address || !seen_bit || !seen_count) {
        fprintf(stderr, "[FAULT INJECTOR] ERROR: Missing required arguments.\n");
        fprintf(stderr, "Usage: -plugin ./fault_injector.so,addr=0x...,bit=...,count=...\n");
    }

    return seen_address & seen_bit & seen_count;
}

// Return TRUE iff it is an ALU instruction with a destination.
static bool IsALUOperation(const char *disas, char *dest_reg, size_t size)
{
    snprintf(dest_reg, size, "NONE");
    
    // 1. Copy disassembly to temp buffer
    char buf[256];
    strncpy(buf, disas, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    // 2. Parse Mnemonic (first word)
    char *mnemonic = buf;
    char *operands = strchr(buf, ' ');
    
    if (operands)
    {
        *operands = '\0'; // Terminate mnemonic
        operands++;       // Point to operands
        while (*operands == ' ') operands++; // Skip spaces
    }

    // ---------------------------------------------------------
    // FILTER: NON-ALU INSTRUCTIONS
    // ---------------------------------------------------------
    
    // 1. NOP (Explicitly skip)
    if (strcmp(mnemonic, "nop") == 0) return false;

    // 2. Loads (Memory Read)
    // Exclude: 'li' (Load Imm), 'lui' (Load Upper Imm) -> Keep these
    if (mnemonic[0] == 'l')
    {
        // Reject: lb, lh, lw, ld, lbu, lhu, lwu, lr.w, lr.d
        if (mnemonic[1] == 'b' || mnemonic[1] == 'h' || mnemonic[1] == 'w' || mnemonic[1] == 'd' || mnemonic[1] == 'r')
             return false;
    }

    // Move Instructions.
    if (strncmp(mnemonic, "mv", 2) == 0) return false; 

    // Float/Vector Loads
    if (strncmp(mnemonic, "fl", 2) == 0) return false; 
    if (strncmp(mnemonic, "vl", 2) == 0) return false;

    // 3. Stores (Memory Write)
    // CAREFUL: sub, sll, srl, sra, slt, sgnj, sqrt are ALU.
    if (mnemonic[0] == 's')
    {
        // Reject: sb, sh, sw, sd (Standard Stores)
        // Matches "sb", "sh", "sw", "sd" exactly
        if (strcmp(mnemonic, "sb") == 0 || strcmp(mnemonic, "sh") == 0 ||
            strcmp(mnemonic, "sw") == 0 || strcmp(mnemonic, "sd") == 0)
            return false;
        
        // Reject Atomics: sc.w, sc.d
        if (strncmp(mnemonic, "sc.", 3) == 0) return false; 
    }

    // Float/Vector Stores
    if (strncmp(mnemonic, "fsw", 3) == 0 || strncmp(mnemonic, "fsd", 3) == 0) return false;
    if (strncmp(mnemonic, "vs", 2) == 0)
    {
        // Heuristic: vs + e/s/x usually store. vsub/vsll are ALU.
        // Reject vse*, vs1r*, vs2r*, vs4r*, vs8r*
        if (mnemonic[2] == 'e' || (mnemonic[2] >= '0' && mnemonic[2] <= '8')) return false;
    }

    // 4. Branches & Jumps
    if (mnemonic[0] == 'b')
    {
        // Reject: beq, bne, blt, bge, ble, bgt
        if (strncmp(mnemonic, "beq", 3) == 0 || strncmp(mnemonic, "bne", 3) == 0 ||
            strncmp(mnemonic, "blt", 3) == 0 || strncmp(mnemonic, "bge", 3) == 0 ||
            strncmp(mnemonic, "ble", 3) == 0 || strncmp(mnemonic, "bgt", 3) == 0) return false;
    }

    // Jumps
    if (mnemonic[0] == 'j') return false; 
    if (strcmp(mnemonic, "call") == 0 || strcmp(mnemonic, "tail") == 0) return false;
    if (strcmp(mnemonic, "ret") == 0) return false;

    if (strncmp(mnemonic, "vs", 2) == 0)
    {
        // vsub, vslidedown, vsll are ALU. vse, vs1r are stores.
        // Heuristic: vs + e/s/x usually store. vsub/vsll are ALU.
        // Safer: Check if it is explicitly a vector store mnemonic if needed.
        // For now, let's assume 'vs' followed by 'e' or '1/2/4/8' is store.
        if (mnemonic[2] == 'e' || (mnemonic[2] >= '0' && mnemonic[2] <= '8')) return false;
    }

    // 5. System / Misc
    if (strncmp(mnemonic, "amo", 3) == 0) return false; 
    if (strncmp(mnemonic, "csr", 3) == 0) return false; 
    if (strcmp(mnemonic, "ecall") == 0 || strcmp(mnemonic, "ebreak") == 0) return false;
    if (strcmp(mnemonic, "fence") == 0 || strcmp(mnemonic, "wfi") == 0) return false;
    if (strcmp(mnemonic, "auipc") == 0) return false;
    if (strcmp(mnemonic, "lui") == 0) return false;

    // ---------------------------------------------------------
    // PARSE DESTINATION (ALU CONFIRMED)
    // ---------------------------------------------------------
    
    // If no operands, return FALSE (It is ALU) but Dest is "NONE"
    if (!operands || *operands == '\0')
        return false; 

    // Extract first token before comma
    char *comma = strchr(operands, ',');
    if (comma)
        *comma = '\0';

    // Check for rounding modes (dyn, rne, rtz, etc.)
    // If found, we must skip this token and treat the NEXT one as the dest.
    if (strcmp(operands, "dyn") == 0 || 
        strcmp(operands, "rne") == 0 || strcmp(operands, "rtz") == 0 ||
        strcmp(operands, "rdn") == 0 || strcmp(operands, "rup") == 0 || 
        strcmp(operands, "rmm") == 0)
    {
        
        // If there was no comma after dyn, we have no dest (e.g. malformed or weird op)
        if (!comma) return false;

        // Advance operands pointer to the next token
        operands = comma + 1;
        while (*operands == ' ') operands++; // Skip spaces

        // Now find the end of THIS new token (the real register)
        comma = strchr(operands, ',');
        if (comma)
            *comma = '\0';
    }

    // Sanity Check: If operand looks like a pure number (e.g. offset), ignore it.
    // Registers usually start with 'a', 'x', 's', 't', 'f', 'v', 'z', 'r', 'g'
    // or 'zero'. If it starts with digit or '-', likely an immediate/offset.
    if ((operands[0] >= '0' && operands[0] <= '9') || operands[0] == '-')
        return true;

    strncpy(dest_reg, operands, size - 1);
    dest_reg[size - 1] = '\0';

    return true;
}

// This is called EVERY TIME a registered instruction executes.
static void InstructionExecutionCallback(unsigned int vcpu_index, void *userdata)
{
    // Only proceed if fault not yet injected.
    if (fault_injected) return;

    // Increment execution count and check if it matches target.
    if ((++current_exec_count) == target_exec_count)
    {
        // Inject fault here.
        fault_injected = true;

        // Get register handle.
        struct qemu_plugin_register* target_register_handle = GetRegisterHandle();
        if (!target_register_handle)
        {
            fprintf(stderr, "[FAULT INJECTOR] ERROR: could not find register handle for %s\n", target_register_name);
            exit(1);
        }
        
        printf("[FAULT INJECTOR] Injecting fault at execution %lu of instruction at 0x%lx\n",
                current_exec_count, target_address);

        // Read current register value
        GByteArray *reg_value = g_byte_array_new();
        int size = qemu_plugin_read_register(target_register_handle, reg_value);

        // If read failed, error out.
        if (size <= 0)
        {
            fprintf(stderr, "[FAULT INJECTOR] ERROR: Could not read register %s\n", target_register_name);
            exit(1);
        }

        // If bit position is out of range, error out.
        if (target_bit_position >= (reg_value->len * 8))
        {
            fprintf(stderr, "[FAULT INJECTOR] ERROR: Bit position %lu out of range for register %s of size %d bytes\n",
                    target_bit_position, target_register_name, reg_value->len);
            exit(1);
        }
        
        // Flip the target bit.
        size_t byte_index = target_bit_position / 8;
        size_t bit_index = target_bit_position % 8;

        uint8_t original_value = reg_value->data[byte_index];
        uint8_t new_value = original_value ^ (1 << bit_index);
        reg_value->data[byte_index] = new_value;
        
        if (qemu_plugin_write_register(target_register_handle, reg_value) != size)
        {
            fprintf(stderr, "[FAULT INJECTOR] ERROR: Could not write modified value to register %s\n", target_register_name);
            exit(1);
        }

        g_byte_array_free(reg_value, TRUE);

        printf("[FAULT INJECTOR] Modified Byte %lu from %s: 0x%02x -> 0x%02x\n", byte_index, target_register_name, original_value, new_value);

        fprintf(output_file, "FAULT_INJECTED\n");
        fprintf(output_file, "Address: 0x%lx\n", target_address);
        fprintf(output_file, "Execution: %lu\n", target_exec_count);
        fprintf(output_file, "Bit: %lu\n", target_bit_position);
        fprintf(output_file, "ByteIndex: %lu\n", byte_index);
        fprintf(output_file, "BitIndex: %lu\n", bit_index);
        fprintf(output_file, "Register: %s\n", target_register_name);
        fprintf(output_file, "OldValue: 0x%02x\n", original_value);
        fprintf(output_file, "NewValue: 0x%02x\n", new_value);
    }
}

// This is called every time a block is translated (once per block)
static void BlockTranslationCallback(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    // Only proceed if we have not found the target instruction.
    if (!target_found)
    {
        size_t n_instructions = qemu_plugin_tb_n_insns(tb);
    
        // Loop through all RISC-V instructions in the block
        for (size_t i = 0; i < n_instructions; i++)
        {
            struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
            const Address_t address = qemu_plugin_insn_vaddr(insn);

            if (address == target_address)
            {
                // Mark that we found the target instruction.
                target_found = true;

                // Get disassembly.
                char disassembly[256];
                char *disas = qemu_plugin_insn_disas(insn);
                CompactSpacesCopy(disas, disassembly);
                g_free(disas);

                // Get symbol (routine name).
                const char *symbol = qemu_plugin_insn_symbol(insn);
                
                if (!IsALUOperation(disassembly, target_register_name, sizeof(target_register_name)))
                {
                    fprintf(stderr, "[FAULT INJECTOR] ERROR: instruction is not ALU at 0x%lx: %s\n", address, disassembly);
                    exit(1);
                }

                if (!IsValidRoutine(symbol))
                {
                    fprintf(stderr, "[FAULT INJECTOR] ERROR: invalid routine for instruction at 0x%lx: %s\n", address, disassembly);
                    exit(1);
                }

                printf("[FAULT INJECTOR] Target instruction found at 0x%lx: %s (Routine: %s), Target Register: %s\n",
                        address, disassembly, symbol ? symbol : "UNKNOWN", target_register_name);

                // Register execution callback for this instruction
                qemu_plugin_register_vcpu_insn_exec_after_cb(
                    insn,
                    InstructionExecutionCallback,
                    QEMU_PLUGIN_CB_RW_REGS,
                    NULL
                );

                break;
            }
        }
    }
}

// Don't forget to cleanup on plugin unload
static void plugin_cleanup(qemu_plugin_id_t id, void *userdata)
{
    printf("[FAULT INJECTOR] Wrote results to %s\n", output_filename);

    if (!fault_injected)
    {
        fprintf(output_file, "NO_FAULT_INJECTED\n");
        fprintf(output_file, "TargetOffset: 0x%lx\n", target_address);
        fprintf(output_file, "TargetExecutionCount: %lu\n", target_exec_count);
        fprintf(output_file, "ActualExecs: %lu\n", current_exec_count);
    }

    fclose(output_file);
}

// Plugin installation
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    // Parse the input arguments.
    ParseArguments(argc, argv);
    fprintf(stderr, "[FAULT INJECTOR] Loaded with: addr=0x%lx, bit=%zu, count=%zu\n", 
            target_address, target_bit_position, target_exec_count);

    // Open output file
    output_file = fopen(output_filename, "w");
    if (output_file == NULL)
    {
        fprintf(stderr, "[FAULT INJECTOR] ERROR: Could not open output file %s\n", output_filename);
        return -1;
    }

    // Initialize.
    current_exec_count = 0;
    fault_injected = false;

    // Register Block Translation Callback.
    qemu_plugin_register_vcpu_tb_trans_cb(id, BlockTranslationCallback);

    // Register cleanup callback on plugin exit.
    qemu_plugin_register_atexit_cb(id, plugin_cleanup, NULL);

    return 0;
}
