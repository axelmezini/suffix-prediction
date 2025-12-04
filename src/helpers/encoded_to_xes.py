import xml.etree.ElementTree as ET
import xml.dom.minidom


def hot_encoded_to_xes(input_file, output_file="neg_traces.xes"):
    """
    Reads a file with traces represented as hot-encoded events and creates an XES file.

    :param input_file: Path to the file containing the hot-encoded traces.
    :param output_file: Path to save the generated XES file.
    """
    # Map hot-encoded events to their corresponding symbols
    encoding_map = {
        (1, 0, 0): "a",
        (0, 1, 0): "b",
        (0, 0, 1): "c",
        (0, 0, 0): "d",
        (1, 1, 0): "d",
        (1, 0, 1): "d",
        (0, 1, 1): "d",
        (1, 1, 1): "d"
    }

    traces = []

    # Read the input file and decode traces
    with open(input_file, "r") as file:
        for line in file:
            events = line.strip().split(";")
            decoded_trace = []
            for event in events:
                hot_encoded = tuple(map(int, event.split(",")))
                symbol = encoding_map.get(hot_encoded)
                if symbol:
                    decoded_trace.append(symbol)
                else:
                    raise ValueError(f"Invalid hot-encoded event: {event}")
            traces.append(decoded_trace)

    # Create the root element for the XES file
    xes = ET.Element("xes", {
        "xmlns": "http://www.xes-standard.org/",
        "xes.version": "2.0",
        "xes.features": "nested-attributes",
        "openxes.version": "1.0RC7",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.xes-standard.org/ http://www.xes-standard.org/xes.xsd"
    })

    # Add global declarations for traces and events
    global_trace = ET.SubElement(xes, "global", {"scope": "trace"})
    ET.SubElement(global_trace, "string", {"key": "concept:name", "value": ""})

    global_event = ET.SubElement(xes, "global", {"scope": "event"})
    ET.SubElement(global_event, "string", {"key": "concept:name", "value": ""})
    ET.SubElement(global_event, "string", {"key": "lifecycle:transition", "value": "complete"})

    # Add classifier
    ET.SubElement(xes, "classifier", {"name": "MXML Legacy Classifier", "keys": "concept:name lifecycle:transition"})

    # Add traces and events
    for trace_id, trace in enumerate(traces):
        trace_element = ET.SubElement(xes, "trace")
        ET.SubElement(trace_element, "string", {"key": "concept:name", "value": f"Trace {trace_id}"})

        for event in trace:
            event_element = ET.SubElement(trace_element, "event")
            ET.SubElement(event_element, "string", {"key": "concept:name", "value": event})
            ET.SubElement(event_element, "string", {"key": "lifecycle:transition", "value": "complete"})

    # Convert the ElementTree to a string
    raw_xml = ET.tostring(xes, encoding="utf-8")

    # Prettify the XML with minidom
    dom = xml.dom.minidom.parseString(raw_xml)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Write the prettified XML to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(pretty_xml)


# Example usage
if __name__ == "__main__":
    input_file = "negative.trace"  # Replace with your input file
    hot_encoded_to_xes(input_file)
