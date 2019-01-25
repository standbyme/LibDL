package com.jstarcraft.core.utility;

import java.net.URL;
import java.net.URLClassLoader;
import java.util.HashMap;
import java.util.Map;

/**
 * 内存类装载器
 * 
 * @author Birdy
 *
 */
public class MemoryClassLoader extends URLClassLoader {

	private Map<String, byte[]> bytes = new HashMap<String, byte[]>();

	public MemoryClassLoader(Map<String, byte[]> bytes) {
		super(new URL[0], Thread.currentThread().getContextClassLoader());
		this.bytes.putAll(bytes);
	}

	@Override
	protected Class<?> findClass(String name) throws ClassNotFoundException {
		byte[] buffer = bytes.get(name);
		if (buffer == null) {
			return super.findClass(name);
		}
		bytes.remove(name);
		return super.defineClass(name, buffer, 0, buffer.length);
	}

}
