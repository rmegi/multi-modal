class ResponseType:
    description: str
    command: str


def parse_response(response: str) -> ResponseType:
    # Check if the response contains a '$' to separate command and description
    if "$" in response:
        # Split at the first occurrence of '$', and include the part that follows as part of the command
        command, description = response.split("$", 1)
        command = f"${command.strip()}{description.split(' ', 1)[0].strip()}"  # Include the full command with the '$'
        description = description[
            len(command) - 1 :
        ].strip()  # Remove the command part from description
    else:
        # If no '$' exists, treat the whole response as description
        command = ""
        description = response.strip()

    # Create and return a ResponseType object
    response_type = ResponseType()
    response_type.command = command
    response_type.description = description
    print(
        f"Parsed response: command='{response_type.command}', description='{response_type.description}'"
    )
    return response_type
