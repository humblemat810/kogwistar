import logging
import os
retry_failed_refine = False
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
ocr_json_version = "0.1"
import time
import base64
from graph_knowledge_engine.engine_core.models import NonText_box_2d, OCRClusterResponse, SplitPage, SplitPageMeta, NonTextCluster, TextCluster
from typing import Any, cast, Callable, Optional,  Literal, TypeAlias, Union
import json
from pydantic_extension.model_slicing import (ModeSlicingMixin, NotMode, FrontendField, BackendField, LLMField,
                DtoType,
                BackendType,
                FrontendType,
                LLMType,
                use_mode)
from pydantic_extension.model_slicing.mixin import ExcludeMode, DtoField
from pydantic import BaseModel, Field, model_validator, field_validator, field_serializer
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
PastCompatibleSplitPage: TypeAlias = SplitPage
def get_page_json(folder_path, page_num):
    with open(os.path.join(folder_path, 'page_'+str(page_num)+'.json'), 'r') as f:
        file_json_raw = json.load(f)
    return file_json_raw
def regen_page(file_json_raw, use_raw):
        # add compatible to union if want to compatible with past models
    """regen from json returned by SplitPage.to_doc(), can be view as SplitPage.FromJson(filepath)"""
    p = PastCompatibleSplitPage(**file_json_raw)
    if use_raw:
        return p.dump_supercede_parse()
    try:
        res = p.to_doc()
    except:
        raise
    return res
def regen_doc(folder_path, use_raw = False):
    pages_nums = sorted((int(i.rsplit(".json",1)[0].split("page_",1)[1]) for i in os.listdir(folder_path) if i.endswith('.json') and i.startswith("page_")))
    pages = []
    split_pages = []
    for pn in pages_nums:
        try:
            pages.append(get_page_json(folder_path, pn))
            split_pages.append(regen_page(pages[-1], use_raw = use_raw))
        except Exception as e:
            folder_path,pn
            print(f'error at page {pn}')
            print(f'in file {folder_path}')
            logger.error(f'error at page {pn}')
            logger.error(f'in file {folder_path}')
            raise
    
    # pages = map( partial(get_page_json, folder_path= folder_path), pages_nums)
    # split_pages = map(regen_page, pages)
    full_doc = list(split_pages)
    return full_doc

class box_2d(BaseModel):
    box_2d: list[int] = Field(description = 'box y min, x min, y max and x max')
    label : str = Field(description = 'text in the box')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')   
class RawOCRResponse(BaseModel):
    """id/ cluster number must be all unique, for example, one of the ocr boxes_2d used id='1', the first image box id (cluster numebr) will be '2', the next signature will be '3' """
    boxes_2d : list[box_2d] = Field(description = 'description of x min, y min, xmax and y max. Share id uniqueness with signature blocks. ')
    non_text_objects:  DtoType[list[NonText_box_2d]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    is_empty_page: DtoType[Optional[bool]] = Field(default = False, description="true if the whole page is empty without recognisable text.")
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
    page_x_min : DtoType[float]=Field(description='the page x min in pixel coordinate. ')
    page_x_max : DtoType[float]=Field(description='the page x max in pixel coordinate. ')
    page_y_min : DtoType[float]=Field(description='the page y min in pixel coordinate. ')
    page_y_max : DtoType[float]=Field(description='the page y max in pixel coordinate. ')
    estimated_rotation_degrees : DtoType[float]=Field(description='the page estimated rotation degree using right hand rule. ')
    incomplete_words_on_edge: DtoType[bool] = Field(description='If there is any text being incomplete due to the scan does not scan the edges properly. ')
    incomplete_text: DtoType[bool]  = Field(description='Any incomplete text')
    data_loss_likelihood: DtoType[float] = Field(description='The likelihood (range from 0.0 to 1.0 inclusive) that the page has lost information by missing the scan data on the edges of the page.' )
    scan_quality: DtoType[Literal['low', 'medium', 'high']] = Field(description='The image quality of the scan. All qualities exclude signatures. '
                                                                                      '"low", "medium" or "high". '
                                'low: text barely legible. medium: Legible with non smooth due to pixelation. high: texts are easily and highly identifiable. ' )
    contains_table: DtoType[bool] = Field(description='Whether this page contains table. ')
    # is_signature_page: DtoType[bool]  = Field(description='Whether this page contains signature. Must agree with signature_blocks')
    # signature_blocks : DtoType[list[SignatureInfo]] = Field(default = [], 
    #                                                         description="The text cluster that belongs to signatory/signature (if any). "
    #                                                         "Indicate whether it is signed or unsigned signatory. "
    #                                                         "Share id uniqueness with OCR text boxes_2d. ")
    
    @model_validator(mode='after')
    def check_cluster_meaningful_ordering_agreement(self):
        assert bool(self.is_empty_page) ^ (len(self.boxes_2d) > 0), f"is_empty_page value {self.is_empty_page} disagree with OCR_text_clusters len={len(self.boxes_2d)}"
        if not len([i.id for i in (self.non_text_objects + self.boxes_2d)]) == len(set(i.id for i in self.non_text_objects + self.boxes_2d)):
            raise ValueError("cluster number from non_text_objects block and ocr text blocks must be ALL distinct. ")
        try:
            if not (len(self.meaningful_ordering) == len(set(self.meaningful_ordering))): # <= len(self.OCR_text_clusters)):
                raise ValueError("meaningful_order must cover each text cluster at most once")
        except Exception as e:
            raise e
        return self


class OCRMetaResponse(BaseModel):
    "meatada of an OCR page"
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    
    page_x_min : DtoType[float]=Field(description='the page x min in pixel coordinate. ')
    page_x_max : DtoType[float]=Field(description='the page x max in pixel coordinate. ')
    page_y_min : DtoType[float]=Field(description='the page y min in pixel coordinate. ')
    page_y_max : DtoType[float]=Field(description='the page y max in pixel coordinate. ')
    estimated_rotation_degrees : DtoType[float]=Field(description='the page estimated rotation degree using right hand rule. ')
    incomplete_words_on_edge: DtoType[bool] = Field(description='If there is any text being incomplete due to the scan does not scan the edges properly. ')
    incomplete_text: DtoType[bool]  = Field(description='Any incomplete text')
    data_loss_likelihood: DtoType[float] = Field(description='The likelihood (range from 0.0 to 1.0 inclusive) that the page has lost information by missing the scan data on the edges of the page.' )
    scan_quality: DtoType[Literal['low', 'medium', 'high']] = Field(description='The image quality of the scan. All qualities exclude signatures. '
                                                                                    '"low", "medium" or "high". '
                                'low: text barely legible. medium: Legible with non smooth due to pixelation. high: texts are easily and highly identifiable. ' )
    contains_table: DtoType[bool] = Field(description='Whether this page contains table. ')
class OCRClusterResponseMetaless(ModeSlicingMixin, BaseModel):
    "response of OCR once meta is quickly skimmed/ determined in separate run"
    OCR_text_clusters: DtoType[list[TextCluster]] = Field(description="the OCR text results.")
    non_text_objects:  DtoType[list[NonTextCluster]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    printed_page_number: DtoType[Optional[str]] = Field(description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
class RawOCRResponseMetaless(ModeSlicingMixin, BaseModel):
    boxes_2d : list[box_2d] = Field(description = 'description of x min, y min, xmax and y max')
    non_text_objects:  DtoType[list[NonText_box_2d]] = Field(description="the non-OCR object results. Share cluster number uniqueness with OCR texts. ")
    printed_page_number: DtoType[Optional[str]] = Field("",description='the page number identified from OCR texts, can be in form of roman numerals such as "i", "ii", "iii", "iv"...; ' 
                                    'Arabic numeral such as 1, 2, 3... or letter such as "a", "b", "c"...\n'
                                    'Sometimes the are surrounded by symbols such as "- 1 -", "- 2 -"'
                                    r"Can be null/none if there is no page order assigned and printed and found in the scanned texts. Do not assign page number. Only use page number found.")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")

def get_first_round_response(draft_responses, llm, model_name, cb, messages, sys_message, img_message, usage_metadata):
    
                    chain = llm.with_structured_output(RawOCRResponse, include_raw = True)
                    before_parse = chain.steps[0]
                    after_parse = chain.steps[1]
                    raw_response = before_parse.invoke(messages, config={"callbacks": [cb]}
                                            )
                    
                    
                    if hasattr(raw_response,"usage_metadata"):
                        usage_metadata.append(raw_response.usage_metadata)
                    else:
                        usage_metadata.append(None)
                    response_with_raw: dict[str, RawOCRResponse] = after_parse.invoke(raw_response)
                    response: RawOCRResponse | OCRClusterResponse | None
                    response1 : RawOCRResponse | None = response_with_raw.get('parsed')
                    raw = response_with_raw.get('raw')
                    parsing_error = response_with_raw.get('parsing_error')
                    if response1 is not None:
                        response = RawOCRResponse_to_OCRClusterResponse(response1)
                    else:
                        response = response1
                    if (response is None) or parsing_error:
                        
                        sys_message_2 = sys_message.model_copy(deep = True)
                        class OCRDraftResponse(BaseModel):
                            text: str = Field(description = "OCR identified text with layout")
                        ocr_draft_response = cast(
                                                    OCRDraftResponse | None,
                                                    llm.with_structured_output(OCRDraftResponse).invoke(messages),
                                                )
                        if (ocr_draft_response is not None) and (ocr_draft_response.text is not None) and ocr_draft_response.text != "":
                            draft_responses[model_name] = ocr_draft_response.text
                        sys_message_2.content += ("If your internal OCR fails. Focus on table parsing mode because my error analysis modes often show that the failing OCR pages are usually highly complicated tables. "
                                                 f"Try to put in as much data as possible given all text found by simple OCR for your reference:```{ocr_draft_response.text}```"  if ocr_draft_response else"")
                        response_with_raw = cast(dict[str, RawOCRResponse], llm.with_structured_output(RawOCRResponse, include_raw = True).invoke(
                            [sys_message, img_message]
                        ))
                        raw_ocr_response = cast(RawOCRResponse, response_with_raw.get('parsed'))
                        if raw_ocr_response is not None:
                            response = RawOCRResponse_to_OCRClusterResponse(raw_ocr_response)
                        raw = response_with_raw.get('raw')
                        parsing_error = response_with_raw.get('parsing_error')
                    return response
def validate_response(response: OCRClusterResponse | None, response_dict: dict,  image_file_path, model_name, page_file_name):
    
                    if response is None:
                        logger.error(f"LLM returned None as response, file name = {image_file_path}, {model_name=}")
                        raise(ValueError(f"LLM returned None as response, file name = {image_file_path}, {model_name=}"))
                    else:
                        if len(response.OCR_text_clusters) == 0 or len(''.join([c.text for c in response.OCR_text_clusters])) == 0:
                            logger.info(f'emptydoc by {model_name}')
                            if model_name != "gemini-2.5-pro": # only trust the verdict from newest advanced model if nothing detected.
                                raise Exception(f"Empty OCR Page error. No text detected at all by less advanced model {model_name}. "
                                                "Application only trust empty response from advanced model 'gemini-2.5-pro'")
                            else:
                                pass
                        else:
                                
                            pass
                        response_dict_local = response.model_dump()
                        response_dict_local['pdf_page_num'] = page_file_name.rsplit('.',1)[0].rsplit("_",1)[-1]
                        response_dict_local['metadata'] = {"ocr_model_name": model_name, "ocr_datetime" : time.time(), "ocr_json_version": str(ocr_json_version)}
                        response_dict_local['refined_version'] = None
                        sp= SplitPage.model_validate(response_dict_local)
                        if sp is None:
                            sp = SplitPage(**response_dict_local)
                        try:
                            sp.to_doc()
                            response_dict.update(response_dict_local)
                            ok = True
                        except Exception as e:
                            logger.error(f"Generated json fail to reproduce doc, file name = {image_file_path}, {model_name=}")
                            logger.error(e)
                            sp.to_doc()
                            raise(ValueError(f"Generated json fail to reproduce doc, file name = {image_file_path}, {model_name=}"))
                        return sp
class TextBox(BaseModel):
    text: str = Field(description = 'identified text')
    bounding_box : list[int] = Field(description = 'Bounding box of identified text')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')
class NonTextObject(BaseModel):
    text: str = Field(description = 'identified non-OCR object')
    bounding_box : list[int] = Field(description = 'Bounding box of identified non-OCR object')
    id: int  = Field(description = 'id of the text box in the page, autoincrement from 0')
class TextBoxResponse(BaseModel):
    text_blocks: list[TextBox] = Field(description = "bounding boxes and text identified")
    non_text_blocks: list[NonTextObject] = Field(description = "bounding boxes and description of the object identified")
    meaningful_ordering : DtoType[list[int]] = Field(description="The correct meaningful ordering of the identified text clusters. Must cover all OCR_text_clusters once and only once. ")
    printed_page_number: str = Field("",description='the page number identified')    
def RawOCRResponse_to_OCRClusterResponse(raw_response: RawOCRResponse | RawOCRResponseMetaless | TextBoxResponse) -> OCRClusterResponse:        
    temp = raw_response.model_dump()
    boxes_2d = temp.pop('boxes_2d')  # y min x min, y max x max
    temp['OCR_text_clusters'] = [TextCluster.model_validate({"text" : i['label'], 
                                                             "bb_y_min" : i['box_2d'][0],
                                                             "bb_x_min" : i['box_2d'][1],
                                                             "bb_y_max" : i['box_2d'][2],
                                                             "bb_x_max" : i['box_2d'][3],
                                                             "cluster_number" : i['id']}) for i in boxes_2d]
    temp['non_text_objects'] = [NonTextCluster.model_validate({"description" : i['label'], 
                                                             "bb_y_min" : i['box_2d'][0],
                                                             "bb_x_min" : i['box_2d'][1],
                                                             "bb_y_max" : i['box_2d'][2],
                                                             "bb_x_max" : i['box_2d'][3],
                                                             "cluster_number" : i['id']}) for i in boxes_2d]
    return OCRClusterResponse.model_validate(temp)
def final_resort(draft_responses: dict, messages, page_file_name, model_name, image_file_path):
                        """
        One day gemini suddenly cannot run but return a totally different schema, ad hoc code fix to fit the transformed schema and
        break down document reading into 2 tasks, namely meta and ocr and non ocr recognition
    """
                        max_v = ""
                        for k, v in draft_responses.items():
                            if len(v) > len(max_v):
                                max_k = k,
                                max_v = v
                        earlier_partial_ocr = draft_responses.get("gemini-2.5-pro") or draft_responses.get("gemini-2.5-flash") or max_v
                        llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.5-pro",
                                    temperature=0.1,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2,
                                    
                                    # other params...
                                )

                        ocr_meta_response: OCRMetaResponse| None =cast (OCRMetaResponse | None , llm.with_structured_output(OCRMetaResponse).invoke(messages[:2]))
                        if ocr_meta_response is None:
                            raise Exception("model capability cannot even skim meta coarse level information")
                        metadata = [SystemMessage("Earlier steps has already determined the metadata about this document: \n\n" + str(ocr_meta_response.model_dump()))]
                        has_error = False
                        try:
                            response2:RawOCRResponseMetaless | None= cast(RawOCRResponseMetaless | None, 
                                                                          llm.with_structured_output(RawOCRResponseMetaless).invoke(messages[:2] + metadata))
                            if response2 is None:
                                has_error = True
                            else:
                                response = RawOCRResponse_to_OCRClusterResponse(response2)
                        except:
                            has_error = True
                        if has_error:
                            # retry only
                            try:
                                response3:TextBoxResponse | None = cast (TextBoxResponse | None , llm.with_structured_output(TextBoxResponse).invoke(messages[:2]))
                                if response3 is None:
                                    has_error = True
                                    raise(Exception("error when trying TextBoxResponse"))
                                else:
                                    response = RawOCRResponse_to_OCRClusterResponse(response3)

                                    coered_response = OCRClusterResponse(
                                                    printed_page_number = ocr_meta_response.printed_page_number,
                                                    page_x_min = ocr_meta_response.page_x_min,
                                                    page_x_max = ocr_meta_response.page_x_max,
                                                    page_y_min = ocr_meta_response.page_y_min,
                                                    page_y_max = ocr_meta_response.page_y_max,
                                                    estimated_rotation_degrees = ocr_meta_response.estimated_rotation_degrees,
                                                    incomplete_words_on_edge =  ocr_meta_response.incomplete_words_on_edge,
                                                    incomplete_text = ocr_meta_response.incomplete_text,
                                                    data_loss_likelihood = ocr_meta_response.data_loss_likelihood,
                                                    scan_quality = ocr_meta_response.scan_quality,
                                                    contains_table = ocr_meta_response.contains_table,
                                                    
                                                    **response.model_dump()
                                                )
                            except Exception as e:
                                try:
                                    coered_response = OCRClusterResponse(
                                                OCR_text_clusters = [TextCluster(text = earlier_partial_ocr, 
                                                                                bb_x_min  = ocr_meta_response.page_x_min, 
                                                                                bb_y_min = ocr_meta_response.page_y_min, 
                                                                                bb_x_max =ocr_meta_response.page_x_max, 
                                                                                bb_y_max =ocr_meta_response.page_y_max, 
                                                                                cluster_number = 0)],
                                                non_text_objects=[],
                                                meaningful_ordering = [0],
                                                printed_page_number = ocr_meta_response.printed_page_number,
                                                page_x_min = ocr_meta_response.page_x_min,
                                                page_x_max = ocr_meta_response.page_x_max,
                                                page_y_min = ocr_meta_response.page_y_min,
                                                page_y_max = ocr_meta_response.page_y_max,
                                                estimated_rotation_degrees = ocr_meta_response.estimated_rotation_degrees,
                                                incomplete_words_on_edge =  ocr_meta_response.incomplete_words_on_edge,
                                                incomplete_text = ocr_meta_response.incomplete_text,
                                                data_loss_likelihood = ocr_meta_response.data_loss_likelihood,
                                                scan_quality = ocr_meta_response.scan_quality,
                                                contains_table = ocr_meta_response.contains_table,
                                            )
                                    response_dict = coered_response.model_dump()
                                    response_dict['pdf_page_num'] = page_file_name.rsplit('.',1)[0].rsplit("_",1)[-1]
                                    response_dict['metadata'] = {"ocr_model_name": model_name, "ocr_datetime" : time.time(), "ocr_json_version": str(ocr_json_version)}
                                    response_dict['refined_version'] = None
                                    
                                    sp= SplitPage.model_validate(response_dict)
                                    if sp is None:
                                        sp = SplitPage(**response_dict)
                                    try:
                                        sp.to_doc()
                                        ok = True
                                    except:
                                        raise Exception("Validation error response_dict cannot be validate into SplitPage")
                                except:
                                    raise(ValueError(f"All LLM failed and coercing final resort fail, file name = {image_file_path}"))
def TextBoxResponse_to_OCRClusterResponse(raw_response: TextBoxResponse, meta_response: OCRMetaResponse) -> OCRClusterResponse:

        
    temp = meta_response.model_dump()
    temp.update(raw_response.model_dump())
    text_blocks = temp.pop('text_blocks')  # y min x min, y max x max
    temp['OCR_text_clusters'] = [TextCluster.model_validate({"text" : i['text'], 
                                                             "bb_y_min" : i['bounding_box'][0],
                                                             "bb_x_min" : i['bounding_box'][1],
                                                             "bb_y_max" : i['bounding_box'][2],
                                                             "bb_x_max" : i['bounding_box'][3],
                                                             "cluster_number" : i['id']}) for i in text_blocks]
    return OCRClusterResponse.model_validate(temp)
from .utils.langchain import GeminiCostCallbackHandler
def refine_image_response(ok2, response_dict, outfile_name, image_file_path, model_names, cb: GeminiCostCallbackHandler):
    
        # if allow_page_refine and (not preexisting):
            if not response_dict:
                with open(outfile_name, 'r') as f:
                    response_dict = json.load(f)
            if response_dict.get('refined_pipeline_run'):
                return False
            if not retry_failed_refine and response_dict.get("refined_pipeline_failed_reason") is not None:
                return False
            cb.total_input_tokens = response_dict['usage_metadata']["input_tokens"]
            cb.total_output_tokens = response_dict['usage_metadata']["output_tokens"]
            cb.total_cost = response_dict['usage_metadata']["total_cost"]
            cb.usage_history = response_dict['usage_metadata']["usage_history"]
            i_model = [i for i, n in enumerate(model_names) if n.startswith('gemini-2.5')][0]
            error_messages = []
            refined = False
            while not ok2:
                refined = False
                try:
                    
                    model_name = model_names[i_model]
                    if "flash-lite" in model_name:
                        i_model += 1
                        if i_model >= min(len(model_names), 20):
                            logger.error(f"All LLM returned None as response, file name = {image_file_path}")
                            raise(ValueError(f"All LLM returned None as response, file name = {image_file_path}"))
                        continue
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.1,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        
                        # other params...
                    )

                    refined = refine_table_ocr(response_dict, llm = llm, cb = cb, error_messages=error_messages)
                    ok2 = True
                except Exception as e:
                    from .utils.log import safe_format_exception
                    e_prompt = safe_format_exception(e)
                    error_message = SystemMessage("post process error raised:\n"
                                                    f"{e_prompt[-10000:]}"
                                                    )
                    error_messages.append(error_message)
                    i_model += 1
                    refined = False
                    response_dict['refined_pipeline_failed_reason'] = "all model exhaused"
                    if i_model >= min(len(model_names), 20):
                        logger.error(f"All LLM returned None as response, file name = {image_file_path}")
                        ok2 = True # it is still ok even not refined
                        # raise(ValueError(f"All LLM returned None as response, file name = {image_file_path}"))
                finally:
                    if refined:
                        response_dict['refined_version']['usage_metadata'] = cb.model_dump() # or usage_metadata
                        response_dict['refined_pipeline_run'] = True
                        with open(outfile_name, 'w') as f:
                            json.dump(response_dict, f)

                        print(response_dict)
            return refined            
def get_messages(image_file_path):
    
            # Open the image in binary mode and read its content.
            with open(image_file_path, "rb") as image_file:
                image_bytes = image_file.read()

            # Base64-encode the binary data.
            encoded_bytes = base64.b64encode(image_bytes)

            # Convert the encoded bytes to a UTF-8 string (optional, if you need a string representation)
            encoded_str = encoded_bytes.decode('utf-8')

            # Print the Base64-encoded string.
            #print(encoded_str)
            sys_message = SystemMessage("You are a helpful raw document AI that does OCR (Optical Character Reading) that focus on extracting raw document meta. "
                                        "You must include spatial arrangement in the responded text. You must include bounding boxes locations x min, x max, y min and y max."
                                        "The user provided image may contain partial document or some lost information on the edges. Try to recover as much information as possible. "
                                        "The user provided image may contain paragraphs, figures or tables. Coerce your result to comply with the required json output format. "
                                        # "If some error messages already found in other attempts, try to focus on parsing complicated tables. Remove watermarks. "
                                        )
            img_message = HumanMessage(
            content=[
                {"type": "text", "text": "find all text in the attached png file. "
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_str}"},
                },
                ],
            )
            return sys_message, img_message
def ocr_single_image(gemini_key: str, page_file_name, file_name, 
                     folder, # out folder
                     exist_behavior: Literal["ok", "skip", "raise", 'rerun']  = 'skip'):
    ok2 = False # stage 2 ok
    outfile_name = os.path.join(folder,file_name, page_file_name.rsplit('.',1)[0] + '.json')
    if os.path.exists(outfile_name):
        if exist_behavior in ["ok", 'rerun']:
            pass
        elif exist_behavior == "skip":
            pass
            # return
        else:
            raise( PermissionError(f"output file {outfile_name} exists"))

    assert gemini_key.startswith("AIza") # gcp keys
    model_names = [# "gemini-2.0-flash", "gemini-2.0-flash-lite", 
                   "gemini-2.5-flash", 
                   "gemini-2.5-flash-lite", "gemini-2.5-pro",
                #    "gemini-1.5-pro",
                   "gemini-2.5-pro-preview-05-06", "gemini-2.5-pro-preview-03-25", 
                     #"gemini-1.5-pro-latest",   deprecated
                     "gemini-2.5-flash-preview-05-20"#,  "gemini-1.5-flash"
                    #  "gemini-2.5-flash-preview-04-17",
                #    "gemini-1.5-pro-001", "gemini-1.5-pro-002", 
                   #"gemini-2.0-flash-thinking-exp-01-21",
                    #"gemini-1.5-flash", 
                    ]
    draft_responses = {}
    ok = False
    i_model = 0
    usage_metadata = []
    from utils.langchain import get_gemini_callback_cost
    response_dict: dict = {}
    image_file_path: str = os.path.join(folder, file_name, page_file_name)
    with get_gemini_callback_cost() as cb:

        if os.path.exists(outfile_name) and exist_behavior == 'rerun' or not os.path.exists(outfile_name):
            sys_message, img_message = get_messages(image_file_path)
            messages = [sys_message, img_message]
            while not ok:
                model_name = model_names[i_model]
                try:
                    
                    llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.1,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                        
                    )
                    response = get_first_round_response(draft_responses, llm, model_name, cb, messages, sys_message, img_message, usage_metadata)
                    sp = validate_response(response, response_dict, image_file_path, model_name, page_file_name)
                    # chain_new_raw = llm.with_structured_output(RawOCRResponse, include_raw = True)
                except Exception as e:
                    from .utils.log import safe_format_exception
                    e_prompt = safe_format_exception(e)
                    error_message = SystemMessage("post process error raised:\n"
                                                    f"{e_prompt[-10000:]}"
                                                    )
                    messages.append(error_message)
                    i_model += 1
                    if i_model >= min(len(model_names), 20):
                        logger.error(f"All LLM returned None as response, file name = {image_file_path}")
                        final_resort(draft_responses, messages, page_file_name, model_name, image_file_path)
                finally:
                    time.sleep(5)
            assert response_dict, Exception("response_dict unbound")
        
            response_dict['usage_metadata'] = cb.model_dump() # or usage_metadata
            with open(outfile_name, 'w') as f:
                json.dump(response_dict, f)
                ok2 = False
            print(response_dict)
        allow_page_refine = False
        if allow_page_refine:
            refined = refine_image_response(ok2, response_dict, outfile_name, image_file_path, model_names, cb)
            if refined:
                time.sleep(5)
    # response_dict = response.model_dump()
    # response_dict['pdf_page_num'] = page_file_name.rsplit('.',1)[0].rsplit("_",1)[-1]
    
    # response_dict['metadata'] = {"ocr_model_name": model_name, "ocr_datetime" : time.time(), "ocr_json_version": str(ocr_json_version)}
   
OCRRefineResponse: TypeAlias = OCRClusterResponse[DtoField]
def refine_table_ocr(response_dict, llm: BaseChatModel, cb, error_messages):
    if response_dict.get('refined_version'):
        return False
    else:
        response_dict['refined_version'] = None
    sp= SplitPage.model_validate(response_dict)
    if sp.refined_version:
        return False
    if not sp.contains_table:
        return False
    if len(sp.OCR_text_clusters) <5:
        return False
    system_prompt = SystemMessage("You are an OCR data organiser. Your job is to refine the user query that contains existing OCR raw text clusters to become meaningful. \n"
                                    "For example, if a table cell or grid is broken down into multiple rows and was classified into multiple cells, \n"
                                    "combine them all into a single meaningful text cluster. Keep metadata unchanged. Make sure no text is ever lost. \n"
                                    "Include all punctionations, typos. Keep spelling errors. You must retain the text from original documents. \n"
                                    "Beware that if an open quotation or open bracket is in one cluster and combine with another cluster, do keep those quotes or brackets.")
    cluster_prompt = HumanMessage(f"{sp}")
        
    # refine llm here\
    messages = [system_prompt, cluster_prompt] + error_messages
    max_attempt = 1 # nice to  have, reorganise
    for i in range(max_attempt):
        try:
            
            oc_refined_result: OCRRefineResponse
            raw: str
            parsing_error: Exception
            from typing import Dict
            temp: dict = cast(dict, llm.with_structured_output(schema = OCRRefineResponse, include_raw = True).invoke(messages, config={"callbacks": [cb]}))
            (raw, oc_refined_result, parsing_error) = (temp['raw'], temp['parsed'], temp['parsing_error'])
            if parsing_error:
                raise parsing_error
            text_before = [i.text for i in sp.OCR_text_clusters]
            text_after = ' '.join([i.text for i in oc_refined_result.OCR_text_clusters])
            from rapidfuzz import fuzz
            def get_threshold(text_before):
                if len(text_before) < 30:
                    threshold = 100
                elif len(text_before) < 60:
                    threshold = 98
                else:
                    threshold = 95
                return threshold
            is_preserved = [fuzz.partial_ratio(text_after, i) >= get_threshold(i) for i in text_before]
            lost_text = [i for i, preserved in zip(sp.OCR_text_clusters, is_preserved) if not preserved]
            ok = all(tf or (tc.text.strip() == "") for tf, tc in zip(is_preserved, sp.OCR_text_clusters))
            assert ok, f"Some text or punctuations are lost through OCR text grouping, lost text = {str(lost_text)}"
            response_dict['refined_version'] = oc_refined_result.model_dump()
            break
        except Exception as e:
            if i < max_attempt-1:
                messages.append(SystemMessage("error found: " + str(e)))
            else:
                raise
    sp= SplitPage.model_validate(response_dict)
    return True