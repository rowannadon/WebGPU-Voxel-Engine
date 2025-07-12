# Apply patch idempotently - only if not already applied
# This script applies a patch file only if it hasn't been applied before

if(NOT PATCH_FILE)
    message(FATAL_ERROR "PATCH_FILE must be defined")
endif()

if(NOT EXISTS "${PATCH_FILE}")
    message(FATAL_ERROR "Patch file does not exist: ${PATCH_FILE}")
endif()

# Check if patch has already been applied by looking for a marker file
set(PATCH_APPLIED_MARKER "${CMAKE_CURRENT_SOURCE_DIR}/.patch_applied")

if(EXISTS "${PATCH_APPLIED_MARKER}")
    message(STATUS "Patch already applied, skipping...")
    return()
endif()

# Try to apply the patch
find_program(GIT_EXECUTABLE git)
if(GIT_EXECUTABLE)
    # Use git apply if available
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE PATCH_CHECK_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    
    if(PATCH_CHECK_RESULT EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE PATCH_RESULT
        )
        
        if(PATCH_RESULT EQUAL 0)
            # Create marker file to indicate patch was applied
            file(WRITE "${PATCH_APPLIED_MARKER}" "Patch applied successfully")
            message(STATUS "Patch applied successfully")
        else()
            message(FATAL_ERROR "Failed to apply patch")
        endif()
    else()
        message(STATUS "Patch appears to be already applied or not applicable")
        # Create marker file anyway
        file(WRITE "${PATCH_APPLIED_MARKER}" "Patch already applied")
    endif()
else()
    message(WARNING "Git not found, skipping patch application")
    # Create marker file to avoid repeated attempts
    file(WRITE "${PATCH_APPLIED_MARKER}" "Git not available")
endif()