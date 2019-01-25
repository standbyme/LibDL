package com.jstarcraft.core.codec.csv;

import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.lang.reflect.Type;

import org.apache.commons.csv.CSVPrinter;

import com.jstarcraft.core.codec.csv.converter.CsvContext;
import com.jstarcraft.core.codec.csv.converter.CsvConverter;
import com.jstarcraft.core.codec.specification.CodecDefinition;
import com.jstarcraft.core.codec.specification.CodecSpecification;
import com.jstarcraft.core.utility.StringUtility;

public class CsvWriter extends CsvContext {

	private CSVPrinter outputStream;

	public CsvWriter(OutputStream outputStream, CodecDefinition definition) {
		super(definition);
		try {
			OutputStreamWriter buffer = new OutputStreamWriter(outputStream, StringUtility.CHARSET);
			this.outputStream = new CSVPrinter(buffer, FORMAT);
		} catch (Exception exception) {
			throw new RuntimeException(exception);
		}
	}

	public CSVPrinter getOutputStream() {
		return outputStream;
	}
//
//	public void write(Type type, Object value) throws Exception {
//		CodecSpecification specification = CodecSpecification.getSpecification(type);
//		CsvConverter converter = converters.get(specification);
//		converter.writeValueTo(this, type, value);
//	}

}