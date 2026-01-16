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

// Structure to hold instruction information.
struct InstructionInfo {
    Address_t m_offset;
    char m_disassembly[256];
    char m_routine[256];
    char m_destination_register[256];
    bool m_has_destination_reg;
    size_t m_execution_count;
};

// Global hash table to store instruction info mapped by address.
GHashTable* instruction_table = NULL;

// Output file name.
static const char* output_filename = "instruction_profile.csv";
static FILE *output_file = NULL;

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

    *(dest++) = '"';
    
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

    *(dest++) = '"';
    *dest = '\0';
}

// Return TRUE iff it is an ALU instruction with or without a destination.
static bool IsALUOperation(const char *disas, bool *has_dest_reg, char *dest_reg, size_t size)
{
    *has_dest_reg = false;
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
    
    // If no operands, return TRUE (It is ALU) but Dest is "NONE"
    if (!operands || *operands == '\0')
        return true; 

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
        if (!comma) return true;

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

    *has_dest_reg = true;
    strncpy(dest_reg, operands, size - 1);
    dest_reg[size - 1] = '\0';

    return true;
}

// This is called EVERY TIME a registered instruction executes.
static void InstructionExecutionCallback(unsigned int vcpu_index, void *userdata)
{
    Address_t address = *(Address_t*)userdata;
    struct InstructionInfo* info = (struct InstructionInfo*) g_hash_table_lookup(instruction_table, &address);

    if (info)
    {
        info->m_execution_count++;
    }
    else
    {
        fprintf(stderr, "[PROFILER] ERROR: InstructionExecutionCallback - No info found for address 0x%lx\n", address);
        exit(1);
    }
}

// This is called every time a block is translated (once per block)
static void BlockTranslationCallback(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    size_t n_instructions = qemu_plugin_tb_n_insns(tb);
    
    // Loop through all RISC-V instructions in the block
    for (size_t i = 0; i < n_instructions; i++)
    {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);

        const Address_t address = qemu_plugin_insn_vaddr(insn);
        const size_t size = qemu_plugin_insn_size(insn);

        uint8_t code[size];  // Capstone requires at least 4 bytes for RISC-V
        const size_t code_count = qemu_plugin_insn_data(insn, code, size);

        // Check if we have already recorded this instruction.
        struct InstructionInfo* info = (struct InstructionInfo*) g_hash_table_lookup(instruction_table, &address);
        
        if (info)
            continue;

        // Get disassembly.
        char disassembly[256];
        char *disas = qemu_plugin_insn_disas(insn);
        strncpy(disassembly, disas, sizeof(disassembly) - 1);
        disassembly[sizeof(disassembly) - 1] = '\0';
        g_free(disas);

        // Init variables for ALU check.
        bool has_destination_register;
        char destination_register[256];

        // Get symbol (routine name).
        const char *symbol = qemu_plugin_insn_symbol(insn);
        
        if (IsALUOperation(disassembly, &has_destination_register, destination_register, sizeof(destination_register)) &&
            IsValidRoutine(symbol))
        {
            struct InstructionInfo* new_info = g_new(struct InstructionInfo, 1);

            // Fill in instruction info.
            new_info->m_offset = address;
            new_info->m_execution_count = 0;

            // Copy destination register.
            new_info->m_has_destination_reg = has_destination_register;
            strncpy(new_info->m_destination_register, destination_register, sizeof(destination_register) - 1);
            new_info->m_destination_register[sizeof(destination_register) - 1] = '\0';

            // Copy dissassembly while compacting spaces.
            CompactSpacesCopy(disassembly, new_info->m_disassembly);
            
            // Lookup symbol for routine name.
            if (symbol)
            {
                strncpy(new_info->m_routine, symbol, sizeof(new_info->m_routine) - 1);
                new_info->m_routine[sizeof(new_info->m_routine) - 1] = '\0';
            }
            else
            {
                strncpy(new_info->m_routine, "UNKNOWN", sizeof(new_info->m_routine) - 1);
                new_info->m_routine[sizeof(new_info->m_routine) - 1] = '\0';
            }

            Address_t *key = g_new(Address_t, 1);
            *key = address;

            g_hash_table_insert(instruction_table, key, new_info);

            // Register execution callback for this instruction
            qemu_plugin_register_vcpu_insn_exec_cb(
                insn,
                InstructionExecutionCallback,
                QEMU_PLUGIN_CB_NO_REGS,
                key
            );
        }

    }
}

// Don't forget to cleanup on plugin unload
static void plugin_cleanup(qemu_plugin_id_t id, void *userdata)
{
    printf("[PROFILER] Finalizing instruction profiling output to %s\n", output_filename);

    fprintf(output_file, "# Instruction Profiling Results\n");
    fprintf(output_file, "# Format: offset,disassembly,routine,dest_register,execution_count\n");
    
    size_t total_instructions = 0;
    size_t instructions_with_dest_reg = 0;

    GHashTableIter iterator;
    gpointer key_ptr, value_ptr;

    g_hash_table_iter_init(&iterator, instruction_table);

    while (g_hash_table_iter_next(&iterator, &key_ptr, &value_ptr))
    {
        const Address_t *key = (const Address_t *)key_ptr;
        struct InstructionInfo *info = (struct InstructionInfo *)value_ptr;
        
        fprintf(output_file, "0x%lx,%s,%s,%s,%s,%lu\n",
                info->m_offset,
                info->m_disassembly,
                info->m_routine,
                info->m_has_destination_reg ? info->m_destination_register : "NONE",
                info->m_has_destination_reg ? "1" : "0",
                info->m_execution_count);

        if (info->m_has_destination_reg)
            instructions_with_dest_reg++;

        total_instructions++;
    }
    
    fprintf(output_file, "# Total unique instructions: %zu\n", total_instructions);
    fprintf(output_file, "# Instructions with dest register: %zu\n", instructions_with_dest_reg);
    
    g_hash_table_destroy(instruction_table);
    fclose(output_file);
}

// Plugin installation
QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv)
{
    instruction_table = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    output_file = fopen(output_filename, "w");
    if (output_file == NULL)
    {
        fprintf(stderr, "[PROFILER] ERROR: Could not open output file %s\n", output_filename);
        return -1;
    }

    // Register Block Translation Callback.
    qemu_plugin_register_vcpu_tb_trans_cb(id, BlockTranslationCallback);

    // Register cleanup callback on plugin exit.
    qemu_plugin_register_atexit_cb(id, plugin_cleanup, NULL);

    return 0;
}
