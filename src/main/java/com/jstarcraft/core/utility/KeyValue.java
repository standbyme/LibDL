package com.jstarcraft.core.utility;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

/**
 * 键值对
 * 
 * @author Birdy
 *
 * @param <K>
 * @param <V>
 */
public class KeyValue<K, V> {

	/** 键 */
	private K key;

	/** 值 */
	private V value;

	KeyValue() {
	}

	public KeyValue(K key, V value) {
		this.key = key;
		this.value = value;
	}

	/**
	 * 获取键
	 * 
	 * @return
	 */
	public K getKey() {
		return key;
	}

	/**
	 * 设置键
	 * 
	 * @param newKey
	 * @return
	 */
	public K setKey(K newKey) {
		K oldKey = key;
		key = newKey;
		return oldKey;
	}

	/**
	 * 获取值
	 * 
	 * @return
	 */
	public V getValue() {
		return value;
	}

	/**
	 * 设置值
	 * 
	 * @param newValue
	 * @return
	 */
	public V setValue(V newValue) {
		V oldValue = value;
		value = newValue;
		return oldValue;
	}

	@Override
	public boolean equals(Object object) {
		if (this == object)
			return true;
		if (object == null)
			return false;
		if (getClass() != object.getClass())
			return false;
		KeyValue that = (KeyValue) object;
		EqualsBuilder equal = new EqualsBuilder();
		equal.append(this.key, that.key);
		equal.append(this.value, that.value);
		return equal.isEquals();
	}

	@Override
	public int hashCode() {
		HashCodeBuilder hash = new HashCodeBuilder();
		hash.append(key);
		hash.append(value);
		return hash.toHashCode();
	}

	@Override
	public String toString() {
		return "KeyValue [key=" + key + ", value=" + value + "]";
	}
}
