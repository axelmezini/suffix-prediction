import xml.etree.ElementTree as ET
import xml.dom.minidom


def traces_to_xes(traces, output_file="traces.xes"):
    """
    Converts a list of traces into an XES file with indentation.

    :param traces: Set of traces where each trace is a string of events.
    :param output_file: Output file name for the XES file.
    """
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

        for event_id, event in enumerate(trace):
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
