import React, { ReactElement } from "react";
import { Select } from "antd";
import { LANGUAGES } from "../../config";
import { SpeakerInterface } from "../../interfaces";

export default function LanguageSelect({
  value,
  className,
  onChange,
  disabled,
}: {
  value: SpeakerInterface["language"] | null;
  className: string | null;
  onChange: ((lang: SpeakerInterface["language"]) => void) | null;
  disabled: boolean;
}): ReactElement {
  if (value === null && onChange !== null) {
    throw new Error(
      `Invalid props received: value is null and onChange is not null, they both have to be null or both have to be not null ...`
    );
  } else if (value !== null && onChange === null) {
    throw new Error(
      `Invalid props received: value is not null and onChange is null, they both have to be null or both have to be not null ...`
    );
  }
  return (
    <Select
      disabled={disabled}
      value={value}
      className={className}
      onChange={onChange}
    >
      {LANGUAGES.map((el) => (
        <Select.Option value={el.iso6391} key={el.iso6391}>
          {el.name}
        </Select.Option>
      ))}
    </Select>
  );
}

LanguageSelect.defaultProps = {
  className: null,
  disabled: false,
  value: null,
  onChange: null,
};
